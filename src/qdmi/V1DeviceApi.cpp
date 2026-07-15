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
#include "qdmi/DeviceManager.hpp"
#include "qdmi/common/Common.hpp"

#include <qdmi/device.h>
#include <spdlog/spdlog.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

class V1Context final {
public:
  V1Context(const std::filesystem::path& path, const std::string& prefix)
      : library_(openLibrary(path)) {
    if (library_ == nullptr) {
      throw std::runtime_error("Could not open QDMI device library: " +
                               path.string());
    }
    try {
      initialize_ = resolve<decltype(QDMI_device_initialize)>(
          library_, prefix, "device_initialize");
      finalize_ = resolve<decltype(QDMI_device_finalize)>(library_, prefix,
                                                          "device_finalize");
      sessionAlloc_ = resolve<decltype(QDMI_device_session_alloc)>(
          library_, prefix, "device_session_alloc");
      sessionInit_ = resolve<decltype(QDMI_device_session_init)>(
          library_, prefix, "device_session_init");
      sessionFree_ = resolve<decltype(QDMI_device_session_free)>(
          library_, prefix, "device_session_free");
      sessionSetParameter_ =
          resolve<decltype(QDMI_device_session_set_parameter)>(
              library_, prefix, "device_session_set_parameter");
      createJob_ = resolve<decltype(QDMI_device_session_create_device_job)>(
          library_, prefix, "device_session_create_device_job");
      jobFree_ = resolve<decltype(QDMI_device_job_free)>(library_, prefix,
                                                         "device_job_free");
      jobSetParameter_ = resolve<decltype(QDMI_device_job_set_parameter)>(
          library_, prefix, "device_job_set_parameter");
      jobQueryProperty_ = resolve<decltype(QDMI_device_job_query_property)>(
          library_, prefix, "device_job_query_property");
      jobSubmit_ = resolve<decltype(QDMI_device_job_submit)>(
          library_, prefix, "device_job_submit");
      jobCancel_ = resolve<decltype(QDMI_device_job_cancel)>(
          library_, prefix, "device_job_cancel");
      jobCheck_ = resolve<decltype(QDMI_device_job_check)>(library_, prefix,
                                                           "device_job_check");
      jobWait_ = resolve<decltype(QDMI_device_job_wait)>(library_, prefix,
                                                         "device_job_wait");
      jobGetResults_ = resolve<decltype(QDMI_device_job_get_results)>(
          library_, prefix, "device_job_get_results");
      queryDevice_ =
          resolve<decltype(QDMI_device_session_query_device_property)>(
              library_, prefix, "device_session_query_device_property");
      querySite_ = resolve<decltype(QDMI_device_session_query_site_property)>(
          library_, prefix, "device_session_query_site_property");
      queryOperation_ =
          resolve<decltype(QDMI_device_session_query_operation_property)>(
              library_, prefix, "device_session_query_operation_property");
      throwIfError(initialize_(), "Initializing QDMI device library");
      initialized_ = true;
    } catch (...) {
      closeLibrary(library_);
      library_ = nullptr;
      throw;
    }
  }

  ~V1Context() {
    if (initialized_) {
      static_cast<void>(finalize_());
    }
    if (library_ != nullptr) {
      closeLibrary(library_);
    }
  }

  [[nodiscard]] auto openSession(const SessionParameters& parameters,
                                 QDMI_Child_Device child) const
      -> QDMI_Device_Session {
    QDMI_Device_Session session = nullptr;
    throwIfError(sessionAlloc_(&session), "Allocating QDMI device session");
    try {
      const auto set = [this,
                        session](const std::optional<std::string>& value,
                                 const QDMI_Device_Session_Parameter key) {
        if (!value) {
          return;
        }
        const auto result = sessionSetParameter_(
            session, key, value->size() + 1, value->c_str());
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          SPDLOG_INFO("QDMI device session parameter {} is not supported",
                      qdmi::toString(key));
          return;
        }
        throwIfError(result, "Setting QDMI device session parameter " +
                                 std::string(qdmi::toString(key)));
      };
      set(parameters.baseUrl, QDMI_DEVICE_SESSION_PARAMETER_BASEURL);
      set(parameters.token, QDMI_DEVICE_SESSION_PARAMETER_TOKEN);
      if (parameters.authFile) {
        set(parameters.authFile->string(),
            QDMI_DEVICE_SESSION_PARAMETER_AUTHFILE);
      }
      set(parameters.authUrl, QDMI_DEVICE_SESSION_PARAMETER_AUTHURL);
      set(parameters.username, QDMI_DEVICE_SESSION_PARAMETER_USERNAME);
      set(parameters.password, QDMI_DEVICE_SESSION_PARAMETER_PASSWORD);
      set(parameters.custom1, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1);
      set(parameters.custom2, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2);
      set(parameters.custom3, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM3);
      set(parameters.custom4, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM4);
      set(parameters.custom5, QDMI_DEVICE_SESSION_PARAMETER_CUSTOM5);
      if (child != nullptr) {
        throwIfError(sessionSetParameter_(
                         session, QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE,
                         sizeof(QDMI_Child_Device),
                         static_cast<const void*>(&child)),
                     "Selecting QDMI child device");
      }
      throwIfError(sessionInit_(session), "Initializing QDMI device session");
      return session;
    } catch (...) {
      sessionFree_(session);
      throw;
    }
  }

  void closeSession(QDMI_Device_Session session) const noexcept {
    if (session != nullptr) {
      sessionFree_(session);
    }
  }

  [[nodiscard]] auto createJob(QDMI_Device_Session session) const
      -> QDMI_Device_Job {
    QDMI_Device_Job job = nullptr;
    throwIfError(createJob_(session, &job), "Creating QDMI device job");
    return job;
  }
  void freeJob(QDMI_Device_Job job) const noexcept {
    if (job != nullptr) {
      jobFree_(job);
    }
  }
  void submitJob(QDMI_Device_Job job) const {
    throwIfError(jobSubmit_(job), "Submitting QDMI job");
  }
  void cancelJob(QDMI_Device_Job job) const {
    throwIfError(jobCancel_(job), "Canceling QDMI job");
  }
  [[nodiscard]] auto checkJob(QDMI_Device_Job job) const -> QDMI_Job_Status {
    QDMI_Job_Status status{};
    throwIfError(jobCheck_(job, &status), "Checking QDMI job");
    return status;
  }
  [[nodiscard]] auto waitJob(QDMI_Device_Job job, const size_t timeout) const
      -> bool {
    const auto result = jobWait_(job, timeout);
    if (result == QDMI_ERROR_TIMEOUT) {
      return false;
    }
    throwIfError(result, "Waiting for QDMI job");
    return true;
  }
  [[nodiscard]] static auto functions() -> DeviceApi::Functions {
    return {
        .openSession =
            [](const void* context, const SessionParameters& parameters,
               const QDMI_Child_Device child) {
              return static_cast<const V1Context*>(context)->openSession(
                  parameters, child);
            },
        .closeSession =
            [](const void* context,
               const QDMI_Device_Session session) noexcept {
              static_cast<const V1Context*>(context)->closeSession(session);
            },
        .createJob =
            [](const void* context, const QDMI_Device_Session session) {
              return static_cast<const V1Context*>(context)->createJob(session);
            },
        .freeJob =
            [](const void* context, const QDMI_Device_Job job) noexcept {
              static_cast<const V1Context*>(context)->freeJob(job);
            },
        .setJobParameter =
            [](const void* context, const QDMI_Device_Job job,
               const QDMI_Device_Job_Parameter parameter, const size_t size,
               const void* value) {
              return static_cast<const V1Context*>(context)->jobSetParameter_(
                  job, parameter, size, value);
            },
        .queryJobProperty =
            [](const void* context, const QDMI_Device_Job job,
               const QDMI_Device_Job_Property property, const size_t size,
               void* value, size_t* sizeRet) {
              return static_cast<const V1Context*>(context)->jobQueryProperty_(
                  job, property, size, value, sizeRet);
            },
        .submitJob =
            [](const void* context, const QDMI_Device_Job job) {
              static_cast<const V1Context*>(context)->submitJob(job);
            },
        .cancelJob =
            [](const void* context, const QDMI_Device_Job job) {
              static_cast<const V1Context*>(context)->cancelJob(job);
            },
        .checkJob =
            [](const void* context, const QDMI_Device_Job job) {
              return static_cast<const V1Context*>(context)->checkJob(job);
            },
        .waitJob =
            [](const void* context, const QDMI_Device_Job job,
               const size_t timeout) {
              return static_cast<const V1Context*>(context)->waitJob(job,
                                                                     timeout);
            },
        .getJobResult =
            [](const void* context, const QDMI_Device_Job job,
               const QDMI_Job_Result result, const size_t size, void* data,
               size_t* sizeRet) {
              return static_cast<const V1Context*>(context)->jobGetResults_(
                  job, result, size, data, sizeRet);
            },
        .queryDevice =
            [](const void* context, const QDMI_Device_Session session,
               const QDMI_Device_Property property, const size_t size,
               void* value, size_t* sizeRet) {
              return static_cast<const V1Context*>(context)->queryDevice_(
                  session, property, size, value, sizeRet);
            },
        .querySite =
            [](const void* context, const QDMI_Device_Session session,
               const QDMI_Site site, const QDMI_Site_Property property,
               const size_t size, void* value, size_t* sizeRet) {
              return static_cast<const V1Context*>(context)->querySite_(
                  session, site, property, size, value, sizeRet);
            },
        .queryOperation =
            [](const void* context, const QDMI_Device_Session session,
               const QDMI_Operation operation, const size_t numSites,
               const QDMI_Site* sites, const size_t numParams,
               const double* params, const QDMI_Operation_Property property,
               const size_t size, void* value, size_t* sizeRet) {
              return static_cast<const V1Context*>(context)->queryOperation_(
                  session, operation, numSites, sites, numParams, params,
                  property, size, value, sizeRet);
            },
    };
  }

private:
  void* library_ = nullptr;
  bool initialized_ = false;
  decltype(QDMI_device_initialize)* initialize_ = nullptr;
  decltype(QDMI_device_finalize)* finalize_ = nullptr;
  decltype(QDMI_device_session_alloc)* sessionAlloc_ = nullptr;
  decltype(QDMI_device_session_init)* sessionInit_ = nullptr;
  decltype(QDMI_device_session_free)* sessionFree_ = nullptr;
  decltype(QDMI_device_session_set_parameter)* sessionSetParameter_ = nullptr;
  decltype(QDMI_device_session_create_device_job)* createJob_ = nullptr;
  decltype(QDMI_device_job_free)* jobFree_ = nullptr;
  decltype(QDMI_device_job_set_parameter)* jobSetParameter_ = nullptr;
  decltype(QDMI_device_job_query_property)* jobQueryProperty_ = nullptr;
  decltype(QDMI_device_job_submit)* jobSubmit_ = nullptr;
  decltype(QDMI_device_job_cancel)* jobCancel_ = nullptr;
  decltype(QDMI_device_job_check)* jobCheck_ = nullptr;
  decltype(QDMI_device_job_wait)* jobWait_ = nullptr;
  decltype(QDMI_device_job_get_results)* jobGetResults_ = nullptr;
  decltype(QDMI_device_session_query_device_property)* queryDevice_ = nullptr;
  decltype(QDMI_device_session_query_site_property)* querySite_ = nullptr;
  decltype(QDMI_device_session_query_operation_property)* queryOperation_ =
      nullptr;
};
} // namespace

auto makeV1DeviceApi(const std::filesystem::path& library,
                     const std::string& prefix)
    -> std::shared_ptr<const DeviceApi> {
  auto context = std::make_shared<V1Context>(library, prefix);
  return std::make_shared<const DeviceApi>(DeviceApi{
      .context = std::move(context), .functions = V1Context::functions()});
}
} // namespace qdmi::detail
