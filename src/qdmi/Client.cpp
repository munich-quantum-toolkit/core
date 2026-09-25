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
#include "qdmi/common/DeviceConfiguration.hpp"
#include "qdmi/driver/DriverExtension.hpp"

#include "support/DiagnosticFormatting.hpp"

#include "qdmi/client.h"

#include "mlir/Support/LogicalResult.h"

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
#endif

#ifndef MQT_CORE_QDMI_DEFAULT_DRIVER_FILENAME
#error                                                                         \
    "MQT_CORE_QDMI_DEFAULT_DRIVER_FILENAME must name the packaged QDMI driver"
#endif

namespace qdmi {
namespace detail {
mlir::FailureOr<std::vector<std::string>>
parseShots(const std::string_view shots, const size_t numShots) {
  if (numShots == 0) {
    if (!shots.empty()) {
      return qdmi::emitError(QDMI_ERROR_FATAL, "Number of shots mismatch");
    }
    return std::vector<std::string>{};
  }

  std::vector<std::string> parsed;
  parsed.reserve(numShots);
  size_t start = 0;
  while (true) {
    const auto end = shots.find(',', start);
    parsed.emplace_back(shots.substr(start, end - start));
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  if (parsed.size() != numShots) {
    return qdmi::emitError(QDMI_ERROR_FATAL, "Number of shots mismatch");
  }
  return parsed;
}

} // namespace detail

namespace {
template <typename T>
mlir::FailureOr<std::map<std::string, T>>
getSparseResult(const detail::ClientAPI& api, QDMI_Job job, size_t programIndex,
                const QDMI_Job_Result keysResult,
                const QDMI_Job_Result valuesResult,
                const std::string& description, const std::string& valueType,
                const std::string& mismatch) {
  size_t keysSize = 0;
  if (mlir::failed(
          qdmi::checkError(api.job_get_results(job, programIndex, keysResult, 0,
                                               nullptr, &keysSize),
                           "Querying " + description + " keys size"))) {
    return mlir::failure();
  }

  if (keysSize == 0) {
    return std::map<std::string, T>{};
  }

  std::string keys(keysSize, '\0');
  if (mlir::failed(
          qdmi::checkError(api.job_get_results(job, programIndex, keysResult,
                                               keysSize, keys.data(), nullptr),
                           "Querying " + description + " keys"))) {
    return mlir::failure();
  }
  keys.pop_back();

  size_t valuesSize = 0;
  if (mlir::failed(
          qdmi::checkError(api.job_get_results(job, programIndex, valuesResult,
                                               0, nullptr, &valuesSize),
                           "Querying " + description + " values size"))) {
    return mlir::failure();
  }

  if (valuesSize % sizeof(T) != 0) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "Invalid " + description +
                               " values size: not a multiple of " + valueType);
  }

  std::vector<T> values(valuesSize / sizeof(T));
  if (mlir::failed(qdmi::checkError(
          api.job_get_results(job, programIndex, valuesResult, valuesSize,
                              values.data(), nullptr),
          "Querying " + description + " values"))) {
    return mlir::failure();
  }

  /// Parse the comma-separated keys.
  std::map<std::string, T> result;
  if (keys.empty() && values.size() == 1) {
    result[""] = values.front();
    return result;
  }
  std::istringstream keysStream(keys);
  std::string key;
  size_t idx = 0;
  while (std::getline(keysStream, key, ',')) {
    if (idx >= values.size()) {
      return qdmi::emitError(QDMI_ERROR_FATAL, mismatch);
    }
    result[key] = values[idx];
    ++idx;
  }

  if (idx != values.size()) {
    return qdmi::emitError(QDMI_ERROR_FATAL, mismatch);
  }
  return result;
}

#ifdef _WIN32
using LibraryHandle = HMODULE;

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
    -> mlir::FailureOr<std::filesystem::path> {
  if (path.empty()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           "QDMI driver path must not be empty");
  }
  std::error_code error;
  auto normalized = std::filesystem::weakly_canonical(
      std::filesystem::absolute(path, error), error);
  if (error) {
    normalized = std::filesystem::absolute(path, error).lexically_normal();
    if (error) {
      return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT, error.message());
    }
  }
  return normalized;
}

[[nodiscard]] auto packagedDriverPath()
    -> mlir::FailureOr<std::filesystem::path> {
  const auto directory = detail::moduleDirectory(
      reinterpret_cast<const void*>(&packagedDriverPath));
  if (directory.empty()) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "Cannot locate the MQT Core QDMI library");
  }
  auto filename = std::filesystem::path{MQT_CORE_QDMI_DEFAULT_DRIVER_FILENAME};
  for (const auto& candidate : {
           directory / filename,
           directory / "lib" / filename,
           directory / "lib64" / filename,
           directory / "bin" / filename,
           directory.parent_path() / "lib" / filename,
           directory.parent_path() / "lib64" / filename,
           directory.parent_path() / "bin" / filename,
       }) {
    std::error_code error;
    if (std::filesystem::exists(candidate, error)) {
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
    -> mlir::FailureOr<std::filesystem::path> {
  if (config.driverPath) {
    return normalizePath(*config.driverPath);
  }
  if (const auto environment = detail::environment("MQT_CORE_QDMI_DRIVER")) {
    return normalizePath(detail::pathFromString(*environment));
  }
  const auto packaged = packagedDriverPath();
  if (mlir::failed(packaged)) {
    return mlir::failure();
  }
  return packaged->has_parent_path() ? normalizePath(*packaged) : packaged;
}

struct DriverExtension {
  decltype(&MQT_CORE_QDMI_driver_add_manifest_v1) addManifest{};
  decltype(&MQT_CORE_QDMI_driver_registered_device_ids_v1)
      registeredDeviceIds{};
  decltype(&MQT_CORE_QDMI_driver_session_alloc_for_device_v1) allocateSession{};
};

struct LoadedDriver : detail::ClientAPI {
  DriverExtension extension;
};

template <class Function>
[[nodiscard]] auto loadSymbol(LibraryHandle library, const char* name)
    -> mlir::FailureOr<Function> {
  /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  const auto function = reinterpret_cast<Function>(findSymbol(library, name));
  if (function == nullptr) {
    return qdmi::emitError(QDMI_ERROR_FATAL, "QDMI driver is missing symbol " +
                                                 std::string(name));
  }
  return function;
}

[[nodiscard]] auto loadClient(const std::filesystem::path& path)
    -> mlir::FailureOr<std::shared_ptr<const LoadedDriver>> {
  struct DriverCache {
    std::mutex mutex;
    std::map<std::filesystem::path, std::shared_ptr<const LoadedDriver>>
        drivers;
  };
  /// Keep validated drivers available to sessions in global destructors.
  /// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
  static auto& cache = *new DriverCache;
  const std::scoped_lock lock(cache.mutex);
  if (const auto found = cache.drivers.find(path);
      found != cache.drivers.end()) {
    return found->second;
  }
  auto* const library = openLibrary(path);
  if (library == nullptr) {
    return qdmi::emitError(QDMI_ERROR_FATAL, "Cannot load QDMI driver '" +
                                                 detail::pathToString(path) +
                                                 "'");
  }

  const std::shared_ptr<void> owner{
      library,
      [](void* handle) { closeLibrary(static_cast<LibraryHandle>(handle)); }};
  auto api = std::make_shared<LoadedDriver>();
  api->library = owner;
  const auto version =
      loadSymbol<decltype(&QDMI_driver_get_client_abi_version)>(
          library, "QDMI_driver_get_client_abi_version");
  if (mlir::failed(version)) {
    return mlir::failure();
  }
  const auto actualAbi = (*version)();
  if (QDMI_VERSION_MAJOR(actualAbi) !=
          QDMI_VERSION_MAJOR(QDMI_CLIENT_ABI_VERSION) ||
      QDMI_VERSION_MINOR(actualAbi) !=
          QDMI_VERSION_MINOR(QDMI_CLIENT_ABI_VERSION)) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "QDMI driver has incompatible ABI " +
                               std::to_string(QDMI_VERSION_MAJOR(actualAbi)) +
                               "." +
                               std::to_string(QDMI_VERSION_MINOR(actualAbi)));
  }

#define LOAD_CLIENT_SYMBOL(symbol)                                             \
  {                                                                            \
    auto result = loadSymbol<decltype(api->symbol)>(library, "QDMI_" #symbol); \
    if (mlir::failed(result)) {                                                \
      return mlir::failure();                                                  \
    }                                                                          \
    api->symbol = *result;                                                     \
  }
  LOAD_CLIENT_SYMBOL(session_alloc);
  LOAD_CLIENT_SYMBOL(session_init);
  LOAD_CLIENT_SYMBOL(session_free);
  LOAD_CLIENT_SYMBOL(session_set_parameter);
  LOAD_CLIENT_SYMBOL(session_query_session_property);
  LOAD_CLIENT_SYMBOL(device_create_job);
  LOAD_CLIENT_SYMBOL(session_retrieve_job_by_id);
  LOAD_CLIENT_SYMBOL(job_free);
  LOAD_CLIENT_SYMBOL(job_set_parameter);
  LOAD_CLIENT_SYMBOL(job_set_programs);
  LOAD_CLIENT_SYMBOL(job_query_property);
  LOAD_CLIENT_SYMBOL(job_submit);
  LOAD_CLIENT_SYMBOL(job_cancel);
  LOAD_CLIENT_SYMBOL(job_check);
  LOAD_CLIENT_SYMBOL(job_wait);
  LOAD_CLIENT_SYMBOL(job_get_results);
  LOAD_CLIENT_SYMBOL(device_query_device_property);
  LOAD_CLIENT_SYMBOL(device_query_site_property);
  LOAD_CLIENT_SYMBOL(device_query_operation_property);
#undef LOAD_CLIENT_SYMBOL
#define LOAD_EXTENSION(field, symbol)                                          \
  api->extension.field = reinterpret_cast<decltype(api->extension.field)>(     \
      findSymbol(library, #symbol))
  LOAD_EXTENSION(addManifest, MQT_CORE_QDMI_driver_add_manifest_v1);
  LOAD_EXTENSION(registeredDeviceIds,
                 MQT_CORE_QDMI_driver_registered_device_ids_v1);
  LOAD_EXTENSION(allocateSession,
                 MQT_CORE_QDMI_driver_session_alloc_for_device_v1);
#undef LOAD_EXTENSION
  return cache.drivers.emplace(path, std::move(api)).first->second;
}

using SessionGuard =
    std::unique_ptr<QDMI_Session_impl_d, decltype(&QDMI_session_free)>;

mlir::LogicalResult validateSessionAllocation(const int status,
                                              QDMI_Session session) {
  if (session == nullptr &&
      (status == QDMI_SUCCESS || status == QDMI_WARN_GENERAL)) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "The QDMI driver returned a null session");
  }
  return qdmi::checkError(status, "Allocating QDMI session");
}

[[nodiscard]] auto allocateSession(const SessionConfig& config)
    -> mlir::FailureOr<std::shared_ptr<detail::DriverSession>> {
  const auto path = requestedDriverPath(config);
  if (mlir::failed(path)) {
    return mlir::failure();
  }
  const auto loaded = loadClient(*path);
  if (mlir::failed(loaded)) {
    return mlir::failure();
  }
  const auto& api = *loaded;
  QDMI_Session session = nullptr;
  const auto status = api->session_alloc(&session);
  SessionGuard guard{session, api->session_free};
  if (mlir::failed(validateSessionAllocation(status, session))) {
    return mlir::failure();
  }
  auto owner = std::make_shared<detail::DriverSession>(api, session);
  /// Ownership transfers only after the shared allocation succeeds.
  /// NOLINTNEXTLINE(bugprone-unused-return-value)
  guard.release();
  return owner;
}
} // namespace

mlir::LogicalResult
builtin_driver::addManifest(const std::filesystem::path& path) {
  const auto driverPath = packagedDriverPath();
  if (mlir::failed(driverPath)) {
    return mlir::failure();
  }
  const auto loaded = loadClient(*driverPath);
  if (mlir::failed(loaded)) {
    return mlir::failure();
  }
  const auto& driver = *loaded;
  if (driver->extension.addManifest == nullptr) {
    return qdmi::emitError(
        QDMI_ERROR_FATAL,
        "The MQT Core QDMI driver does not support device manifests");
  }
  const auto normalized = normalizePath(path);
  if (mlir::failed(normalized)) {
    return mlir::failure();
  }
  const auto filename = detail::pathToString(*normalized);
  return qdmi::checkError(driver->extension.addManifest(filename.c_str()),
                          "Registering QDMI device manifest");
}

mlir::FailureOr<std::vector<std::string>>
builtin_driver::registeredDeviceIds() {
  const auto path = packagedDriverPath();
  if (mlir::failed(path)) {
    return mlir::failure();
  }
  const auto loaded = loadClient(*path);
  if (mlir::failed(loaded)) {
    return mlir::failure();
  }
  const auto& driver = *loaded;
  const auto query = driver->extension.registeredDeviceIds;
  if (query == nullptr) {
    return qdmi::emitError(
        QDMI_ERROR_FATAL,
        "The MQT Core QDMI driver does not support offline enumeration");
  }
  size_t size = 0;
  if (mlir::failed(qdmi::checkError(query(0, nullptr, &size),
                                    "Querying registered QDMI device IDs"))) {
    return mlir::failure();
  }
  std::string buffer(size, '\0');
  if (size != 0 &&
      mlir::failed(qdmi::checkError(query(size, buffer.data(), nullptr),
                                    "Querying registered QDMI device IDs"))) {
    return mlir::failure();
  }
  std::vector<std::string> ids;
  for (size_t start = 0; start < buffer.size();) {
    const auto end = buffer.find('\0', start);
    if (end == std::string::npos) {
      return qdmi::emitError(QDMI_ERROR_FATAL,
                             "The QDMI driver returned an unterminated ID");
    }
    ids.emplace_back(buffer.substr(start, end - start));
    start = end + 1;
  }
  return ids;
}

mlir::FailureOr<Device> builtin_driver::openDevice(
    const std::string_view id, const std::string_view deviceSessionJson,
    const std::optional<std::filesystem::path>& driverPath) {
  if (id.empty() || id.find('\0') != std::string_view::npos) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "QDMI device ID must not be empty or contain null bytes");
  }
  const auto path =
      driverPath ? normalizePath(*driverPath) : packagedDriverPath();
  if (mlir::failed(path)) {
    return mlir::failure();
  }
  const auto loaded = loadClient(*path);
  if (mlir::failed(loaded)) {
    return mlir::failure();
  }
  const auto& driver = *loaded;
  if (driver->extension.allocateSession == nullptr) {
    return qdmi::emitError(
        QDMI_ERROR_FATAL, "The QDMI driver does not support targeted sessions");
  }
  QDMI_Session session = nullptr;
  const std::string deviceId{id};
  const auto status = driver->extension.allocateSession(
      deviceId.c_str(), deviceSessionJson.size(),
      deviceSessionJson.empty() ? nullptr : deviceSessionJson.data(), &session);
  SessionGuard guard{session, driver->session_free};
  if (mlir::failed(validateSessionAllocation(status, session))) {
    return mlir::failure();
  }
  auto owner = std::make_shared<detail::DriverSession>(driver, session);
  /// NOLINTNEXTLINE(bugprone-unused-return-value)
  guard.release();
  if (mlir::failed(qdmi::checkError(driver->session_init(session),
                                    "Initializing QDMI device session"))) {
    return mlir::failure();
  }
  size_t size = 0;
  if (mlir::failed(qdmi::checkError(
          driver->session_query_session_property(
              session, QDMI_SESSION_PROPERTY_DEVICES, 0, nullptr, &size),
          "Querying QDMI device"))) {
    return mlir::failure();
  }
  if (size != sizeof(QDMI_Device)) {
    return qdmi::emitError(
        QDMI_ERROR_FATAL,
        "A targeted QDMI session must expose exactly one device");
  }
  QDMI_Device device = nullptr;
  if (mlir::failed(
          qdmi::checkError(driver->session_query_session_property(
                               session, QDMI_SESSION_PROPERTY_DEVICES, size,
                               static_cast<void*>(&device), nullptr),
                           "Querying QDMI device"))) {
    return mlir::failure();
  }
  if (device == nullptr) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "A targeted QDMI session returned a null device");
  }
  return Device{device, std::move(owner)};
}

void detail::JobDeleter::operator()(QDMI_Job_impl_d* const job) const {
  if (job != nullptr) {
    session->api->job_free(job);
  }
}

mlir::FailureOr<size_t> Site::getIndex() const {
  return queryProperty<size_t>(QDMI_SITE_PROPERTY_INDEX);
}
mlir::FailureOr<std::optional<uint64_t>> Site::getT1() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T1);
}
mlir::FailureOr<std::optional<uint64_t>> Site::getT2() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_T2);
}
mlir::FailureOr<std::optional<std::string>> Site::getName() const {
  return queryProperty<std::optional<std::string>>(QDMI_SITE_PROPERTY_NAME);
}
mlir::FailureOr<std::optional<int64_t>> Site::getXCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_XCOORDINATE);
}
mlir::FailureOr<std::optional<int64_t>> Site::getYCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_YCOORDINATE);
}
mlir::FailureOr<std::optional<int64_t>> Site::getZCoordinate() const {
  return queryProperty<std::optional<int64_t>>(QDMI_SITE_PROPERTY_ZCOORDINATE);
}
mlir::FailureOr<bool> Site::isZone() const {
  auto result = queryProperty<std::optional<bool>>(QDMI_SITE_PROPERTY_ISZONE);
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  return (*result).value_or(false);
}
mlir::FailureOr<std::optional<uint64_t>> Site::getXExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_XEXTENT);
}
mlir::FailureOr<std::optional<uint64_t>> Site::getYExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_YEXTENT);
}
mlir::FailureOr<std::optional<uint64_t>> Site::getZExtent() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_ZEXTENT);
}
mlir::FailureOr<std::optional<uint64_t>> Site::getModuleIndex() const {
  return queryProperty<std::optional<uint64_t>>(QDMI_SITE_PROPERTY_MODULEINDEX);
}
mlir::FailureOr<std::optional<uint64_t>> Site::getSubmoduleIndex() const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_SITE_PROPERTY_SUBMODULEINDEX);
}
mlir::FailureOr<std::string>
Operation::getName(const std::vector<Site>& sites,
                   const std::vector<double>& params) const {
  return queryProperty<std::string>(QDMI_OPERATION_PROPERTY_NAME, sites,
                                    params);
}
mlir::FailureOr<std::optional<size_t>>
Operation::getQubitsNum(const std::vector<Site>& sites,
                        const std::vector<double>& params) const {
  return queryProperty<std::optional<size_t>>(QDMI_OPERATION_PROPERTY_QUBITSNUM,
                                              sites, params);
}
mlir::FailureOr<size_t>
Operation::getParametersNum(const std::vector<Site>& sites,
                            const std::vector<double>& params) const {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sites,
                               params);
}
mlir::FailureOr<std::optional<uint64_t>>
Operation::getDuration(const std::vector<Site>& sites,
                       const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_DURATION, sites, params);
}
mlir::FailureOr<std::optional<double>>
Operation::getFidelity(const std::vector<Site>& sites,
                       const std::vector<double>& params) const {
  return queryProperty<std::optional<double>>(QDMI_OPERATION_PROPERTY_FIDELITY,
                                              sites, params);
}
mlir::FailureOr<std::optional<uint64_t>>
Operation::getInteractionRadius(const std::vector<Site>& sites,
                                const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS, sites, params);
}
mlir::FailureOr<std::optional<uint64_t>>
Operation::getBlockingRadius(const std::vector<Site>& sites,
                             const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS, sites, params);
}
mlir::FailureOr<std::optional<double>>
Operation::getIdlingFidelity(const std::vector<Site>& sites,
                             const std::vector<double>& params) const {
  return queryProperty<std::optional<double>>(
      QDMI_OPERATION_PROPERTY_IDLINGFIDELITY, sites, params);
}
mlir::FailureOr<bool> Operation::isZoned() const {
  auto result = queryProperty<std::optional<bool>>(
      QDMI_OPERATION_PROPERTY_ISZONED, {}, {});
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  return (*result).value_or(false);
}
mlir::FailureOr<std::optional<std::vector<Site>>> Operation::getSites() const {
  auto qdmiSitesResult = queryProperty<std::optional<std::vector<QDMI_Site>>>(
      QDMI_OPERATION_PROPERTY_SITES, {}, {});
  if (mlir::failed(qdmiSitesResult)) {
    return mlir::failure();
  }
  auto& qdmiSites = (*qdmiSitesResult);
  if (!qdmiSites.has_value()) {
    return std::optional<std::vector<Site>>{};
  }
  std::vector<Site> returnedSites;
  returnedSites.reserve(qdmiSites->size());
  std::ranges::transform(*qdmiSites, std::back_inserter(returnedSites),
                         [this](const QDMI_Site& site) -> Site {
                           return {device_, session_, site};
                         });
  return std::optional<std::vector<Site>>{std::move(returnedSites)};
}
mlir::FailureOr<std::optional<std::vector<std::pair<Site, Site>>>>
Operation::getSitePairs() const {
  auto qubitsNum = getQubitsNum({}, {});
  if (mlir::failed(qubitsNum)) {
    return mlir::failure();
  }
  if ((*qubitsNum) != 2) {
    return std::optional<std::vector<std::pair<Site, Site>>>{};
  }
  auto zoned = isZoned();
  if (mlir::failed(zoned)) {
    return mlir::failure();
  }
  if ((*zoned)) {
    return std::optional<std::vector<std::pair<Site, Site>>>{};
  }
  auto sites = getSites();
  if (mlir::failed(sites)) {
    return mlir::failure();
  }
  const auto& sitesOpt = (*sites);
  if (!sitesOpt || sitesOpt->empty() || sitesOpt->size() % 2 != 0) {
    return std::optional<std::vector<std::pair<Site, Site>>>{};
  }
  std::vector<std::pair<Site, Site>> pairs;
  pairs.reserve(sitesOpt->size() / 2);
  for (size_t i = 0; i < sitesOpt->size(); i += 2) {
    pairs.emplace_back((*sitesOpt)[i], (*sitesOpt)[i + 1]);
  }
  return std::optional<std::vector<std::pair<Site, Site>>>{std::move(pairs)};
}
mlir::FailureOr<std::optional<uint64_t>>
Operation::getMeanShuttlingSpeed(const std::vector<Site>& sites,
                                 const std::vector<double>& params) const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED, sites, params);
}
mlir::FailureOr<std::string> Device::getId() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_ID);
}
mlir::FailureOr<std::string> Device::getName() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_NAME);
}

mlir::FailureOr<std::string> Device::getVersion() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_VERSION);
}

mlir::FailureOr<QDMI_Device_Status> Device::getStatus() const {
  return queryProperty<QDMI_Device_Status>(QDMI_DEVICE_PROPERTY_STATUS);
}

mlir::FailureOr<std::string> Device::getLibraryVersion() const {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_LIBRARYVERSION);
}

mlir::FailureOr<size_t> Device::getQubitsNum() const {
  return queryProperty<size_t>(QDMI_DEVICE_PROPERTY_QUBITSNUM);
}

mlir::FailureOr<std::vector<Site>> Device::getSites() const {
  auto qdmiSitesResult =
      queryProperty<std::vector<QDMI_Site>>(QDMI_DEVICE_PROPERTY_SITES);
  if (mlir::failed(qdmiSitesResult)) {
    return mlir::failure();
  }
  auto& qdmiSites = (*qdmiSitesResult);
  std::vector<Site> sites;
  sites.reserve(qdmiSites.size());
  std::ranges::transform(qdmiSites, std::back_inserter(sites),
                         [this](const QDMI_Site& site) -> Site {
                           return {device_, session_, site};
                         });
  return sites;
}

mlir::FailureOr<std::vector<Site>> Device::getRegularSites() const {
  auto result = getSites();
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  std::vector<Site> sites;
  for (auto& site : (*result)) {
    auto zone = site.isZone();
    if (mlir::failed(zone)) {
      return mlir::failure();
    }
    if (!(*zone)) {
      sites.emplace_back(std::move(site));
    }
  }
  return sites;
}

mlir::FailureOr<std::vector<Site>> Device::getZones() const {
  auto result = getSites();
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  std::vector<Site> sites;
  for (auto& site : (*result)) {
    auto zone = site.isZone();
    if (mlir::failed(zone)) {
      return mlir::failure();
    }
    if ((*zone)) {
      sites.emplace_back(std::move(site));
    }
  }
  return sites;
}

mlir::FailureOr<std::vector<Operation>> Device::getOperations() const {
  auto qdmiOperationsResult = queryProperty<std::vector<QDMI_Operation>>(
      QDMI_DEVICE_PROPERTY_OPERATIONS);
  if (mlir::failed(qdmiOperationsResult)) {
    return mlir::failure();
  }
  auto& qdmiOperations = (*qdmiOperationsResult);
  return wrapOperations(qdmiOperations);
}

mlir::FailureOr<std::optional<std::vector<Operation>>>
Device::queryCustomOperations(const CustomProperty property) const {
  auto propertyResult = detail::toDeviceProperty(property);
  if (mlir::failed(propertyResult)) {
    return mlir::failure();
  }
  const auto qdmiProperty = (*propertyResult);
  auto handlesResult = detail::queryHandleArray<QDMI_Operation>(
      [this, qdmiProperty](const size_t size, void* value, size_t* sizeRet) {
        return api().device_query_device_property(device_, qdmiProperty, size,
                                                  value, sizeRet);
      },
      "custom operation list " +
          std::to_string(static_cast<unsigned>(property)));
  if (mlir::failed(handlesResult)) {
    return mlir::failure();
  }
  auto& handles = (*handlesResult);
  if (!handles.has_value()) {
    return std::optional<std::vector<Operation>>{};
  }
  return std::optional<std::vector<Operation>>{
      std::move(wrapOperations(*handles))};
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

mlir::FailureOr<std::optional<std::vector<std::pair<Site, Site>>>>
Device::getCouplingMap() const {
  auto qdmiCouplingMapResult = queryProperty<
      std::optional<std::vector<std::pair<QDMI_Site, QDMI_Site>>>>(
      QDMI_DEVICE_PROPERTY_COUPLINGMAP);
  if (mlir::failed(qdmiCouplingMapResult)) {
    return mlir::failure();
  }
  auto& qdmiCouplingMap = (*qdmiCouplingMapResult);
  if (!qdmiCouplingMap.has_value()) {
    return std::optional<std::vector<std::pair<Site, Site>>>{};
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
  return std::optional<std::vector<std::pair<Site, Site>>>{
      std::move(couplingMap)};
}

mlir::FailureOr<std::optional<size_t>> Device::getQueueLength() const {
  return queryProperty<std::optional<size_t>>(QDMI_DEVICE_PROPERTY_QUEUELENGTH);
}

mlir::FailureOr<std::optional<std::string>> Device::getLengthUnit() const {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_LENGTHUNIT);
}

mlir::FailureOr<std::optional<double>> Device::getLengthScaleFactor() const {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR);
}

mlir::FailureOr<std::optional<std::string>> Device::getDurationUnit() const {
  return queryProperty<std::optional<std::string>>(
      QDMI_DEVICE_PROPERTY_DURATIONUNIT);
}

mlir::FailureOr<std::optional<double>> Device::getDurationScaleFactor() const {
  return queryProperty<std::optional<double>>(
      QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR);
}

mlir::FailureOr<std::optional<uint64_t>> Device::getMinAtomDistance() const {
  return queryProperty<std::optional<uint64_t>>(
      QDMI_DEVICE_PROPERTY_MINATOMDISTANCE);
}

mlir::FailureOr<std::vector<QDMI_Program_Format>>
Device::getSupportedProgramFormats() const {
  return queryProperty<std::vector<QDMI_Program_Format>>(
      QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS);
}

mlir::FailureOr<std::vector<Device>> Device::getChildDevices() const {
  size_t size = 0;
  auto result = api().device_query_device_property(
      device_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr, &size);
  if (result == QDMI_ERROR_NOTSUPPORTED) {
    return std::vector<Device>{};
  }
  if (mlir::failed(checkError(result, "Querying child devices size"))) {
    return mlir::failure();
  }
  if (size % sizeof(QDMI_Device) != 0) {
    return qdmi::emitError(QDMI_ERROR_FATAL, "Invalid child device list size");
  }

  std::vector<QDMI_Device> handles(size / sizeof(QDMI_Device));
  if (size != 0) {
    result = api().device_query_device_property(
        device_, QDMI_DEVICE_PROPERTY_CHILDDEVICES, size,
        static_cast<void*>(handles.data()), nullptr);
    if (mlir::failed(checkError(result, "Querying child devices"))) {
      return mlir::failure();
    }
  }

  std::vector<Device> devices;
  devices.reserve(handles.size());
  std::ranges::transform(handles, std::back_inserter(devices),
                         [this](QDMI_Device_impl_d* const handle) {
                           return Device(handle, session_);
                         });
  return devices;
}

mlir::FailureOr<Job>
Device::submitJob(const std::string& program, const QDMI_Program_Format format,
                  const size_t numShots,
                  const std::optional<CustomJobParameter>& custom1,
                  const std::optional<CustomJobParameter>& custom2,
                  const std::optional<CustomJobParameter>& custom3,
                  const std::optional<CustomJobParameter>& custom4,
                  const std::optional<CustomJobParameter>& custom5) const {
  if (isBinaryProgramFormat(format)) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Binary program formats require exact-byte submission");
  }

  return submitPrograms({&program, 1}, format, numShots, custom1, custom2,
                        custom3, custom4, custom5);
}

mlir::FailureOr<Job>
Device::submitJob(const std::string& program, const QDMI_Program_Format format,
                  const std::optional<CustomJobParameter>& custom1,
                  const std::optional<CustomJobParameter>& custom2,
                  const std::optional<CustomJobParameter>& custom3,
                  const std::optional<CustomJobParameter>& custom4,
                  const std::optional<CustomJobParameter>& custom5) const {
  if (isBinaryProgramFormat(format)) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Binary program formats require exact-byte submission");
  }

  return submitPrograms({&program, 1}, format, std::nullopt, custom1, custom2,
                        custom3, custom4, custom5);
}

mlir::FailureOr<Job>
Device::submitJob(const std::span<const std::byte> program,
                  const QDMI_Program_Format format, const size_t numShots,
                  const std::optional<CustomJobParameter>& custom1,
                  const std::optional<CustomJobParameter>& custom2,
                  const std::optional<CustomJobParameter>& custom3,
                  const std::optional<CustomJobParameter>& custom4,
                  const std::optional<CustomJobParameter>& custom5) const {

  return submitPrograms({&program, 1}, format, numShots, custom1, custom2,
                        custom3, custom4, custom5);
}

mlir::FailureOr<Job>
Device::submitJob(const std::span<const std::byte> program,
                  const QDMI_Program_Format format,
                  const std::optional<CustomJobParameter>& custom1,
                  const std::optional<CustomJobParameter>& custom2,
                  const std::optional<CustomJobParameter>& custom3,
                  const std::optional<CustomJobParameter>& custom4,
                  const std::optional<CustomJobParameter>& custom5) const {

  return submitPrograms({&program, 1}, format, std::nullopt, custom1, custom2,
                        custom3, custom4, custom5);
}

mlir::FailureOr<Job>
Device::submitPrograms(const std::span<const std::string> programs,
                       const QDMI_Program_Format format,
                       const std::optional<size_t> numShots,
                       const std::optional<CustomJobParameter>& custom1,
                       const std::optional<CustomJobParameter>& custom2,
                       const std::optional<CustomJobParameter>& custom3,
                       const std::optional<CustomJobParameter>& custom4,
                       const std::optional<CustomJobParameter>& custom5) const {
  auto job = trySubmitPrograms(programs, format, numShots, custom1, custom2,
                               custom3, custom4, custom5);
  if (mlir::failed(job)) {
    return mlir::failure();
  }
  if (!*job) {
    return qdmi::emitError(QDMI_ERROR_NOTSUPPORTED, "Setting programs");
  }
  return std::move(**job);
}

mlir::FailureOr<std::optional<Job>> Device::trySubmitPrograms(
    const std::span<const std::string> programs,
    const QDMI_Program_Format format, const std::optional<size_t> numShots,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  if (isBinaryProgramFormat(format)) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Binary program formats require exact-byte submission");
  }
  std::vector<size_t> sizes;
  std::vector<const void*> pointers;
  sizes.reserve(programs.size());
  pointers.reserve(programs.size());
  for (const auto& program : programs) {
    const auto terminator = program.find('\0');
    if (terminator != std::string::npos && terminator != program.size() - 1) {
      return qdmi::emitError(
          QDMI_ERROR_INVALIDARGUMENT,
          "Text programs must not contain embedded null bytes");
    }
    sizes.push_back(program.size() + (terminator == std::string::npos ? 1 : 0));
    pointers.push_back(program.c_str());
  }
  return submitProgramsImpl(format, sizes, pointers, numShots, custom1, custom2,
                            custom3, custom4, custom5);
}

mlir::FailureOr<Job> Device::submitPrograms(
    const std::span<const std::span<const std::byte>> programs,
    const QDMI_Program_Format format, const std::optional<size_t> numShots,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  auto job = trySubmitPrograms(programs, format, numShots, custom1, custom2,
                               custom3, custom4, custom5);
  if (mlir::failed(job)) {
    return mlir::failure();
  }
  if (!*job) {
    return qdmi::emitError(QDMI_ERROR_NOTSUPPORTED, "Setting programs");
  }
  return std::move(**job);
}

mlir::FailureOr<std::optional<Job>> Device::trySubmitPrograms(
    const std::span<const std::span<const std::byte>> programs,
    const QDMI_Program_Format format, const std::optional<size_t> numShots,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  std::vector<size_t> sizes;
  std::vector<const void*> pointers;
  sizes.reserve(programs.size());
  pointers.reserve(programs.size());
  for (const auto& program : programs) {
    sizes.push_back(program.size());
    pointers.push_back(program.data());
  }
  return submitProgramsImpl(format, sizes, pointers, numShots, custom1, custom2,
                            custom3, custom4, custom5);
}

mlir::FailureOr<std::optional<Job>> Device::submitProgramsImpl(
    const QDMI_Program_Format format, const std::span<const size_t> sizes,
    const std::span<const void* const> programs,
    const std::optional<size_t> numShots,
    const std::optional<CustomJobParameter>& custom1,
    const std::optional<CustomJobParameter>& custom2,
    const std::optional<CustomJobParameter>& custom3,
    const std::optional<CustomJobParameter>& custom4,
    const std::optional<CustomJobParameter>& custom5) const {
  QDMI_Job job = nullptr;
  if (mlir::failed(qdmi::checkError(api().device_create_job(device_, &job),
                                    "Creating job"))) {
    return mlir::failure();
  }
  Job jobWrapper{job, session_};
  if (mlir::failed(qdmi::checkError(
          api().job_set_parameter(jobWrapper, QDMI_JOB_PARAMETER_PROGRAMFORMAT,
                                  sizeof(format), &format),
          "Setting program format"))) {
    return mlir::failure();
  }
  if (numShots.has_value()) {
    if (mlir::failed(qdmi::checkError(
            api().job_set_parameter(jobWrapper, QDMI_JOB_PARAMETER_SHOTSNUM,
                                    sizeof(*numShots), &*numShots),
            "Setting number of shots"))) {
      return mlir::failure();
    }
  }

  if (custom1.has_value()) {
    if (mlir::failed(setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM1,
                                       *custom1))) {
      return mlir::failure();
    }
  }
  if (custom2.has_value()) {
    if (mlir::failed(setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM2,
                                       *custom2))) {
      return mlir::failure();
    }
  }
  if (custom3.has_value()) {
    if (mlir::failed(setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM3,
                                       *custom3))) {
      return mlir::failure();
    }
  }
  if (custom4.has_value()) {
    if (mlir::failed(setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM4,
                                       *custom4))) {
      return mlir::failure();
    }
  }
  if (custom5.has_value()) {
    if (mlir::failed(setCustomJobParam(jobWrapper, QDMI_JOB_PARAMETER_CUSTOM5,
                                       *custom5))) {
      return mlir::failure();
    }
  }

  const auto result = api().job_set_programs(
      jobWrapper, &format, programs.size(), sizes.data(), programs.data());
  if (result == QDMI_ERROR_NOTSUPPORTED) {
    return std::optional<Job>{};
  }
  if (mlir::failed(qdmi::checkError(result, "Setting programs"))) {
    return mlir::failure();
  }
  if (mlir::failed(
          qdmi::checkError(api().job_submit(jobWrapper), "Submitting job"))) {
    return mlir::failure();
  }
  return std::optional<Job>{std::move(jobWrapper)};
}

mlir::FailureOr<Job>
Device::retrieveJobById(const std::string_view jobId) const {
  const std::string id{jobId};
  QDMI_Job job = nullptr;
  if (mlir::failed(qdmi::checkError(
          api().session_retrieve_job_by_id(device_, id.c_str(), &job),
          "Retrieving job"))) {
    return mlir::failure();
  }
  return Job{job, session_};
}

mlir::LogicalResult
Device::setCustomJobParam(QDMI_Job job, const QDMI_Job_Parameter param,
                          const CustomJobParameter& value) const {
  return std::visit(
      [&]<typename CustomValue>(const CustomValue& customValue) {
        using T = std::decay_t<CustomValue>;
        if constexpr (std::is_same_v<T, std::string>) {
          if (mlir::failed(qdmi::checkError(
                  api().job_set_parameter(job, param, customValue.size() + 1,
                                          customValue.c_str()),
                  "Setting custom parameter"))) {
            return mlir::failure();
          }
        } else {
          static_assert(std::is_trivially_copyable_v<T>,
                        "Custom job parameters must be trivially copyable");
          if (mlir::failed(qdmi::checkError(
                  api().job_set_parameter(job, param, sizeof(T), &customValue),
                  "Setting custom parameter"))) {
            return mlir::failure();
          }
        }
        return mlir::success();
      },
      value);
}

mlir::FailureOr<QDMI_Job_Status> Job::check() const {
  QDMI_Job_Status status{};
  if (mlir::failed(qdmi::checkError(api().job_check(job_.get(), &status),
                                    "Checking job status"))) {
    return mlir::failure();
  }
  return status;
}

mlir::FailureOr<bool> Job::wait(const size_t timeout) const {
  const auto ret = api().job_wait(job_.get(), timeout);
  if (ret == QDMI_SUCCESS) {
    return true;
  }
  if (ret == QDMI_ERROR_TIMEOUT) {
    return false;
  }
  if (mlir::failed(checkError(ret, "Waiting for job"))) {
    return mlir::failure();
  }
  return true;
}

mlir::LogicalResult Job::cancel() const {
  return qdmi::checkError(api().job_cancel(job_.get()), "Cancelling job");
}

mlir::FailureOr<std::string> Job::getId() const {
  return detail::queryProperty<std::string>(
      [this](const size_t size, void* value, size_t* sizeRet) {
        return api().job_query_property(job_.get(), QDMI_JOB_PROPERTY_ID, size,
                                        value, sizeRet);
      },
      "Querying job ID", "Querying job ID size");
}

mlir::FailureOr<QDMI_Program_Format> Job::getProgramFormat() const {
  QDMI_Program_Format format{};
  if (mlir::failed(qdmi::checkError(
          api().job_query_property(job_.get(), QDMI_JOB_PROPERTY_PROGRAMFORMAT,
                                   sizeof(format), &format, nullptr),
          "Querying program format"))) {
    return mlir::failure();
  }
  return format;
}

mlir::FailureOr<std::vector<std::byte>> Job::getProgramBytes() const {
  size_t size = 0;
  if (mlir::failed(qdmi::checkError(
          api().job_query_property(job_.get(), QDMI_JOB_PROPERTY_PROGRAM, 0,
                                   nullptr, &size),
          "Querying program size"))) {
    return mlir::failure();
  }

  std::vector<std::byte> program(size);
  if (size != 0 &&
      mlir::failed(qdmi::checkError(
          api().job_query_property(job_.get(), QDMI_JOB_PROPERTY_PROGRAM, size,
                                   program.data(), nullptr),
          "Querying program"))) {
    return mlir::failure();
  }

  return program;
}

mlir::FailureOr<std::string> Job::getProgram() const {
  auto formatResult = getProgramFormat();
  if (mlir::failed(formatResult)) {
    return mlir::failure();
  }
  auto const& format = (*formatResult);
  if (isBinaryProgramFormat(format)) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           "Cannot decode a binary program as a string; use "
                           "getProgramBytes()");
  }

  auto programResult = getProgramBytes();
  if (mlir::failed(programResult)) {
    return mlir::failure();
  }
  auto& program = (*programResult);
  if (program.empty() || program.back() != std::byte{0}) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "Cannot decode program as a null-terminated string; use "
        "getProgramBytes() for binary payloads");
  }
  return std::string(reinterpret_cast<const char*>(program.data()),
                     program.size() - 1);
}

mlir::FailureOr<size_t> Job::getNumShots() const {
  size_t numShots = 0;
  if (mlir::failed(qdmi::checkError(
          api().job_query_property(job_.get(), QDMI_JOB_PROPERTY_SHOTSNUM,
                                   sizeof(numShots), &numShots, nullptr),
          "Querying number of shots"))) {
    return mlir::failure();
  }
  return numShots;
}

mlir::FailureOr<size_t> Job::getNumPrograms() const {
  size_t count = 0;
  if (mlir::failed(qdmi::checkError(
          api().job_query_property(job_.get(), QDMI_JOB_PROPERTY_PROGRAMSNUM,
                                   sizeof(count), &count, nullptr),
          "Querying program count"))) {
    return mlir::failure();
  }
  return count;
}

mlir::FailureOr<std::optional<std::vector<QDMI_Job_Status>>>
Job::getProgramStatuses() const {
  auto statuses =
      detail::queryProperty<std::optional<std::vector<QDMI_Job_Status>>>(
          [&](size_t size, void* value, size_t* sizeRet) {
            return api().job_query_property(job_.get(),
                                            QDMI_JOB_PROPERTY_PROGRAMSTATUSES,
                                            size, value, sizeRet);
          },
          "Querying program statuses", "Querying program status size");
  if (mlir::failed(statuses)) {
    return mlir::failure();
  }
  if (!*statuses) {
    return statuses;
  }
  const auto count = getNumPrograms();
  if (mlir::failed(count)) {
    return mlir::failure();
  }
  if ((**statuses).size() != *count) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "Program status count does not match the job");
  }
  return statuses;
}

mlir::FailureOr<std::vector<std::byte>>
Job::getResults(const QDMI_Job_Result result, const size_t programIndex) const {
  size_t size = 0;
  if (mlir::failed(
          qdmi::checkError(api().job_get_results(job_.get(), programIndex,
                                                 result, 0, nullptr, &size),
                           "Querying result size"))) {
    return mlir::failure();
  }
  std::vector<std::byte> value(size);
  if (size != 0 && mlir::failed(qdmi::checkError(
                       api().job_get_results(job_.get(), programIndex, result,
                                             size, value.data(), nullptr),
                       "Querying result"))) {
    return mlir::failure();
  }
  return value;
}

mlir::FailureOr<std::optional<size_t>> Job::getQueuePosition() const {
  size_t queuePosition = 0;
  const auto result =
      api().job_query_property(job_.get(), QDMI_JOB_PROPERTY_QUEUEPOSITION,
                               sizeof(queuePosition), &queuePosition, nullptr);
  return detail::queuePositionFromResult(result, queuePosition);
}

mlir::FailureOr<std::vector<std::string>>
Job::getShots(const size_t programIndex) const {
  size_t shotsSize = 0;
  if (mlir::failed(qdmi::checkError(
          api().job_get_results(job_.get(), programIndex, QDMI_JOB_RESULT_SHOTS,
                                0, nullptr, &shotsSize),
          "Querying shots size"))) {
    return mlir::failure();
  }

  if (shotsSize == 0) {
    return std::vector<std::string>{};
  }

  std::string shots(shotsSize, '\0');
  if (mlir::failed(qdmi::checkError(
          api().job_get_results(job_.get(), programIndex, QDMI_JOB_RESULT_SHOTS,
                                shotsSize, shots.data(), nullptr),
          "Querying shots"))) {
    return mlir::failure();
  }
  shots.pop_back();

  auto count = getNumShots();
  if (mlir::failed(count)) {
    return mlir::failure();
  }
  return detail::parseShots(shots, (*count));
}

mlir::FailureOr<std::map<std::string, size_t>>
Job::getCounts(const size_t programIndex) const {
  return getSparseResult<size_t>(
      api(), job_.get(), programIndex, QDMI_JOB_RESULT_HIST_KEYS,
      QDMI_JOB_RESULT_HIST_VALUES, "histogram", "size_t",
      "Histogram key/value count mismatch");
}

mlir::FailureOr<std::vector<std::complex<double>>>
Job::getDenseStateVector(const size_t programIndex) const {
  size_t size = 0;
  if (mlir::failed(qdmi::checkError(
          api().job_get_results(job_.get(), programIndex,
                                QDMI_JOB_RESULT_STATEVECTOR_DENSE, 0, nullptr,
                                &size),
          "Querying dense state vector size"))) {
    return mlir::failure();
  }

  if (size % sizeof(std::complex<double>) != 0) {
    return qdmi::emitError(
        QDMI_ERROR_FATAL,
        "Invalid state vector size: not a multiple of complex<double>");
  }

  std::vector<std::complex<double>> stateVector(size /
                                                sizeof(std::complex<double>));
  if (mlir::failed(qdmi::checkError(
          api().job_get_results(job_.get(), programIndex,
                                QDMI_JOB_RESULT_STATEVECTOR_DENSE, size,
                                stateVector.data(), nullptr),
          "Querying dense state vector"))) {
    return mlir::failure();
  }
  return stateVector;
}

mlir::FailureOr<std::vector<double>>
Job::getDenseProbabilities(const size_t programIndex) const {
  size_t size = 0;
  if (mlir::failed(qdmi::checkError(
          api().job_get_results(job_.get(), programIndex,
                                QDMI_JOB_RESULT_PROBABILITIES_DENSE, 0, nullptr,
                                &size),
          "Querying dense probabilities size"))) {
    return mlir::failure();
  }

  if (size % sizeof(double) != 0) {
    return qdmi::emitError(
        QDMI_ERROR_FATAL,
        "Invalid probabilities size: not a multiple of double");
  }

  std::vector<double> probabilities(size / sizeof(double));
  if (mlir::failed(qdmi::checkError(
          api().job_get_results(job_.get(), programIndex,
                                QDMI_JOB_RESULT_PROBABILITIES_DENSE, size,
                                probabilities.data(), nullptr),
          "Querying dense probabilities"))) {
    return mlir::failure();
  }
  return probabilities;
}

mlir::FailureOr<std::map<std::string, std::complex<double>>>
Job::getSparseStateVector(const size_t programIndex) const {
  return getSparseResult<std::complex<double>>(
      api(), job_.get(), programIndex, QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS,
      QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES, "sparse state vector",
      "complex<double>", "Sparse state vector key/value count mismatch");
}

mlir::FailureOr<std::map<std::string, double>>
Job::getSparseProbabilities(const size_t programIndex) const {
  return getSparseResult<double>(
      api(), job_.get(), programIndex,
      QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS,
      QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES, "sparse probabilities",
      "double", "Sparse probabilities key/value count mismatch");
}

mlir::FailureOr<Device> Session::openDevice(const std::string_view id,
                                            const SessionConfig& config) {
  auto session = Session::create(config);
  if (mlir::failed(session)) {
    return mlir::failure();
  }
  return session->getDevice(id);
}

mlir::FailureOr<std::vector<std::string>> Session::getDeviceIds() {
  std::vector<std::string> ids;
  const auto devices = getDevices();
  if (mlir::failed(devices)) {
    return mlir::failure();
  }
  for (const auto& device : *devices) {
    const auto id = device.getId();
    if (mlir::failed(id)) {
      return mlir::failure();
    }
    ids.emplace_back(*id);
  }
  return ids;
}

mlir::FailureOr<Device> Session::getDevice(const std::string_view id) {
  if (id.empty() || id.find('\0') != std::string_view::npos) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        "QDMI device ID must not be empty or contain null bytes");
  }
  const auto devices = getDevices();
  if (mlir::failed(devices)) {
    return mlir::failure();
  }
  std::string available;
  for (const auto& device : *devices) {
    const auto candidateId = device.getId();
    if (mlir::failed(candidateId)) {
      return mlir::failure();
    }
    if (*candidateId == id) {
      return device;
    }
    if (!available.empty()) {
      available += ", ";
    }
    available += *candidateId;
  }
  return qdmi::emitError(QDMI_ERROR_OUTOFRANGE,
                         "QDMI driver session has no device with ID '" +
                             std::string(id) +
                             "'; available IDs: " + available);
}

mlir::FailureOr<Session> Session::create(const SessionConfig& config) {
  auto allocated = allocateSession(config);
  if (mlir::failed(allocated)) {
    return mlir::failure();
  }
  Session session;
  session.session_ = std::move(*allocated);
  const auto setParameter =
      [&session](const std::optional<std::string>& value,
                 QDMI_Session_Parameter param) -> mlir::LogicalResult {
    if (!value) {
      return mlir::success();
    }
    const auto status = session.api().session_set_parameter(
        session.session_->handle.get(), param, value->size() + 1U,
        value->c_str());
    if (status == QDMI_ERROR_NOTSUPPORTED) {
      ::mqt::diagnostics::info("Session parameter {} not supported (skipped)",
                               qdmi::toString(param));
      return mlir::success();
    }
    return qdmi::checkError(status, std::string("Setting session parameter ") +
                                        qdmi::toString(param));
  };
  if (mlir::failed(setParameter(config.token, QDMI_SESSION_PARAMETER_TOKEN))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.authUrl, QDMI_SESSION_PARAMETER_AUTHURL))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.username, QDMI_SESSION_PARAMETER_USERNAME))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.password, QDMI_SESSION_PARAMETER_PASSWORD))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.projectId, QDMI_SESSION_PARAMETER_PROJECTID))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.custom1, QDMI_SESSION_PARAMETER_CUSTOM1))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.custom2, QDMI_SESSION_PARAMETER_CUSTOM2))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.custom3, QDMI_SESSION_PARAMETER_CUSTOM3))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.custom4, QDMI_SESSION_PARAMETER_CUSTOM4))) {
    return mlir::failure();
  }
  if (mlir::failed(
          setParameter(config.custom5, QDMI_SESSION_PARAMETER_CUSTOM5))) {
    return mlir::failure();
  }
  if (config.authFile &&
      mlir::failed(setParameter(detail::pathToString(*config.authFile),
                                QDMI_SESSION_PARAMETER_AUTHFILE))) {
    return mlir::failure();
  }
  if (mlir::failed(qdmi::checkError(
          session.api().session_init(session.session_->handle.get()),
          "Initializing session"))) {
    return mlir::failure();
  }
  return session;
}

mlir::FailureOr<std::vector<Device>> Session::getDevices() {
  auto qdmiDevicesResult =
      queryProperty<std::vector<QDMI_Device>>(QDMI_SESSION_PROPERTY_DEVICES);
  if (mlir::failed(qdmiDevicesResult)) {
    return mlir::failure();
  }
  auto& qdmiDevices = (*qdmiDevicesResult);
  std::vector<Device> devices;
  devices.reserve(qdmiDevices.size());
  std::ranges::transform(qdmiDevices, std::back_inserter(devices),
                         [this](QDMI_Device_impl_d* const& dev) -> Device {
                           return {dev, session_};
                         });
  return devices;
}
} // namespace qdmi
