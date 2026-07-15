/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "V1DeviceApi.hpp"

#include "qdmi/common/Common.hpp"

#include <qdmi/device.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qdmi::detail {
namespace {
#ifdef _WIN32
[[nodiscard]] auto openLibrary(const std::filesystem::path& path) -> void* {
  return LoadLibraryExW(path.wstring().c_str(), nullptr,
                        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
}
[[nodiscard]] auto loadSymbol(void* library, const std::string& symbol)
    -> void* {
  return reinterpret_cast<void*>(
      GetProcAddress(static_cast<HMODULE>(library), symbol.c_str()));
}
void closeLibrary(void* library) {
  static_cast<void>(FreeLibrary(static_cast<HMODULE>(library)));
}
#else
[[nodiscard]] auto openLibrary(const std::filesystem::path& path) -> void* {
  return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
}
[[nodiscard]] auto loadSymbol(void* library, const std::string& symbol)
    -> void* {
  return dlsym(library, symbol.c_str());
}
void closeLibrary(void* library) { static_cast<void>(dlclose(library)); }
#endif

template <class Function>
[[nodiscard]] auto resolve(void* library, const std::string& prefix,
                           const std::string& suffix) -> Function* {
  const auto name = prefix + "_QDMI_" + suffix;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* function = reinterpret_cast<Function*>(loadSymbol(library, name));
  if (function == nullptr) {
    throw std::runtime_error("Failed to load QDMI v1 symbol '" + name + "'");
  }
  return function;
}
} // namespace

V1DeviceApi::V1DeviceApi(const std::filesystem::path& library,
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

V1DeviceApi::~V1DeviceApi() {
  if (initialized_) {
    static_cast<void>(finalize_());
  }
  if (library_ != nullptr) {
    closeLibrary(library_);
  }
}

} // namespace qdmi::detail
