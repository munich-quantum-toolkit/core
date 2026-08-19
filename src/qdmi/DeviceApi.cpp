/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "DeviceApi.hpp"

#include "qdmi/common/Common.hpp"

#include <qdmi/device.h>

#include <filesystem>
#include <latch>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qdmi::detail {
namespace {
struct DeviceApiCacheEntry {
  std::weak_ptr<const DeviceApi> api;
  std::shared_ptr<std::latch> finalized;
};

struct FinalizationSignal {
  std::shared_ptr<std::latch> finalized;

  ~FinalizationSignal() { finalized->count_down(); }
};

struct DeviceApiGeneration {
  FinalizationSignal signal;
  DeviceApi api;

  DeviceApiGeneration(const std::filesystem::path& library,
                      const std::string& prefix,
                      std::shared_ptr<std::latch> finalized)
      : signal{std::move(finalized)}, api(library, prefix) {}
};

struct DeviceApiCache {
  std::mutex mutex;
  std::map<std::string, DeviceApiCacheEntry> libraries;
};

[[nodiscard]] DeviceApiCache& deviceApiCache() {
  static DeviceApiCache cache;
  return cache;
}

#ifdef _WIN32
[[nodiscard]] void* openLibrary(const std::filesystem::path& path) {
  return LoadLibraryExW(path.wstring().c_str(), nullptr,
                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
[[nodiscard]] void* loadSymbol(void* library, const std::string& symbol) {
  return reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(library), symbol.c_str()));
}
void closeLibrary(void* library) {
  static_cast<void>(FreeLibrary(static_cast<HMODULE>(library)));
}
#else
[[nodiscard]] void* openLibrary(const std::filesystem::path& path) {
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}
[[nodiscard]] void* loadSymbol(void* library, const std::string& symbol) {
  return dlsym(library, symbol.c_str());
}
void closeLibrary(void* library) { static_cast<void>(dlclose(library)); }
#endif

template <class Function>
[[nodiscard]] Function* resolve(void* library, const std::string& prefix,
                                const std::string& suffix) {
  const auto name = prefix + "_QDMI_" + suffix;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* function = reinterpret_cast<Function*>(loadSymbol(library, name));
  if (function == nullptr) {
    throw std::runtime_error("Failed to load QDMI symbol '" + name + "'");
  }
  return function;
}

template <class Function>
[[nodiscard]] Function* resolveOptional(void* library,
                                        const std::string& prefix,
                                        const std::string& suffix) {
  const auto name = prefix + "_QDMI_" + suffix;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return reinterpret_cast<Function*>(loadSymbol(library, name));
}
} // namespace

DeviceApi::DeviceApi(const std::filesystem::path& library,
                     const std::string& prefix)
    : library_(openLibrary(library)) {
  if (library_ == nullptr) {
    throw std::runtime_error("Could not open QDMI device library: " +
                             library.string());
  }
  try {
    const auto initialize = resolve<decltype(QDMI_device_initialize)>(
        library_, prefix, "device_initialize");
    finalize_ = resolve<decltype(QDMI_device_finalize)>(library_, prefix,
                                                        "device_finalize");
#define LOAD_QDMI_SYMBOL(name)                                                 \
  name = resolve<decltype(QDMI_##name)>(library_, prefix, #name)
    LOAD_QDMI_SYMBOL(device_session_alloc);
    LOAD_QDMI_SYMBOL(device_session_init);
    LOAD_QDMI_SYMBOL(device_session_free);
    LOAD_QDMI_SYMBOL(device_session_set_parameter);
    LOAD_QDMI_SYMBOL(device_session_create_device_job);
    device_session_retrieve_device_job_by_id = resolveOptional<
        decltype(QDMI_device_session_retrieve_device_job_by_id)>(
        library_, prefix, "device_session_retrieve_device_job_by_id");
    LOAD_QDMI_SYMBOL(device_job_free);
    LOAD_QDMI_SYMBOL(device_job_set_parameter);
    LOAD_QDMI_SYMBOL(device_job_query_property);
    LOAD_QDMI_SYMBOL(device_job_submit);
    LOAD_QDMI_SYMBOL(device_job_cancel);
    LOAD_QDMI_SYMBOL(device_job_check);
    LOAD_QDMI_SYMBOL(device_job_wait);
    LOAD_QDMI_SYMBOL(device_job_get_results);
    LOAD_QDMI_SYMBOL(device_session_query_device_property);
    LOAD_QDMI_SYMBOL(device_session_query_site_property);
    LOAD_QDMI_SYMBOL(device_session_query_operation_property);
#undef LOAD_QDMI_SYMBOL
    throwIfError(initialize(), "Initializing QDMI device library");
    initialized_ = true;
  } catch (...) {
    closeLibrary(library_);
    library_ = nullptr;
    throw;
  }
}

DeviceApi::~DeviceApi() {
  if (initialized_) {
    static_cast<void>(finalize_());
  }
  if (library_ != nullptr) {
    closeLibrary(library_);
  }
}

std::shared_ptr<const DeviceApi>
loadDeviceApi(const std::filesystem::path& library, const std::string& prefix) {
  const auto canonicalLibrary = std::filesystem::weakly_canonical(library);
  const auto key = canonicalLibrary.string() + "\n" + prefix;
  auto& cache = deviceApiCache();
  while (true) {
    std::unique_lock lock(cache.mutex);
    if (const auto entry = cache.libraries.find(key);
        entry != cache.libraries.end()) {
      if (auto loaded = entry->second.api.lock()) {
        return loaded;
      }
      const auto finalized = entry->second.finalized;
      lock.unlock();
      finalized->wait();
      lock.lock();
      if (const auto stale = cache.libraries.find(key);
          stale != cache.libraries.end() &&
          stale->second.finalized == finalized) {
        cache.libraries.erase(stale);
      }
      continue;
    }

    auto finalized = std::make_shared<std::latch>(1);
    auto generation = std::make_shared<DeviceApiGeneration>(canonicalLibrary,
                                                            prefix, finalized);
    auto loaded =
        std::shared_ptr<const DeviceApi>(generation, &generation->api);
    cache.libraries.emplace(
        key,
        DeviceApiCacheEntry{.api = loaded, .finalized = std::move(finalized)});
    return loaded;
  }
}

} // namespace qdmi::detail
