/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/common/DeviceConfiguration.hpp"

#include "qdmi/common/Common.hpp"

#include "qdmi/device.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <memory>
#include <windows.h>
#else
#include <dlfcn.h>
#ifdef __linux__
#include <elf.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <vector>
#endif
#endif

namespace qdmi::detail {
llvm::Expected<std::filesystem::path> moduleDirectory(const void* anchor) {
  if (anchor == nullptr) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           "Module anchor must not be null");
  }
#ifdef _WIN32
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(anchor), &module) == 0) {
    return qdmi::makeError(QDMI_ERROR_FATAL, "Cannot locate the QDMI module");
  }
  std::wstring buffer(MAX_PATH, L'\0');
  while (true) {
    const auto length = GetModuleFileNameW(module, buffer.data(),
                                           static_cast<DWORD>(buffer.size()));
    if (length == 0) {
      return qdmi::makeError(QDMI_ERROR_FATAL, "Cannot locate the QDMI module");
    }
    if (length < buffer.size()) {
      buffer.resize(length);
      return std::filesystem::path(buffer).parent_path();
    }
    buffer.resize(buffer.size() * 2);
  }
#else
  Dl_info info{};
  if (dladdr(anchor, &info) == 0 || info.dli_fname == nullptr) {
    return qdmi::makeError(QDMI_ERROR_FATAL, "Cannot locate the QDMI module");
  }
  auto path = std::filesystem::path(info.dli_fname);
  if (!path.is_absolute()) {
#ifdef __linux__
    Dl_info executable{};
    const auto* headers = reinterpret_cast<const void*>(getauxval(AT_PHDR));
    if (dladdr(headers, &executable) != 0 &&
        info.dli_fbase == executable.dli_fbase) {
      std::error_code error;
      path = std::filesystem::read_symlink("/proc/self/exe", error);
      if (error) {
        return qdmi::makeError(QDMI_ERROR_FATAL,
                               "Cannot locate the QDMI module");
      }
    }
#elif defined(__APPLE__)
    if (info.dli_fbase == _dyld_get_image_header(0)) {
      uint32_t size = 0;
      static_cast<void>(_NSGetExecutablePath(nullptr, &size));
      std::vector<char> buffer(size);
      if (_NSGetExecutablePath(buffer.data(), &size) != 0) {
        return qdmi::makeError(QDMI_ERROR_FATAL,
                               "Cannot locate the QDMI module");
      }
      path = buffer.data();
    }
#endif
  }
  std::error_code error;
  const auto canonical = std::filesystem::weakly_canonical(path, error);
  if (error) {
    return qdmi::makeError(QDMI_ERROR_FATAL,
                           "Cannot resolve the QDMI module path: " +
                               error.message());
  }
  return canonical.parent_path();
#endif
}

namespace {
[[nodiscard]] llvm::Expected<LoadedDeviceConfiguration>
readFile(const std::filesystem::path& path, const bool bundled) {
  auto source = qdmi::detail::pathToString(path);
  if (!source) {
    return source.takeError();
  }
  auto buffer = llvm::MemoryBuffer::getFile(*source);
  if (!buffer) {
    int status = QDMI_ERROR_PERMISSIONDENIED;
    if (bundled) {
      status = QDMI_ERROR_FATAL;
    } else if (buffer.getError() == std::errc::no_such_file_or_directory ||
               buffer.getError() == std::errc::not_a_directory) {
      status = QDMI_ERROR_NOTFOUND;
    }
    return qdmi::makeError(
        status,
        llvm::formatv("Cannot read QDMI device configuration '{0}': {1}",
                      *source, buffer.getError().message())
            .str());
  }
  return LoadedDeviceConfiguration{
      .json = (*buffer)->getBuffer().str(),
      .source = std::move(*source),
  };
}
} // namespace

int setDeviceConfigurationParameter(
    const QDMI_Device_Session_Parameter parameter, const size_t size,
    const void* value, std::optional<std::string>& inlineJson,
    std::optional<std::filesystem::path>& file) {
  if (parameter != QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1 &&
      parameter != QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2) {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  if (value == nullptr) {
    return size == 0 ? QDMI_SUCCESS : QDMI_ERROR_INVALIDARGUMENT;
  }
  if (size == 0) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::span bytes{static_cast<const char*>(value), size};
  if (bytes.back() != '\0' ||
      std::memchr(bytes.data(), '\0', bytes.size() - 1) != nullptr) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  const std::string stringValue(bytes.data(), bytes.size() - 1);
  if (parameter == QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1) {
    inlineJson =
        stringValue.empty() ? std::nullopt : std::optional{stringValue};
  } else {
    if (stringValue.empty()) {
      file.reset();
    } else {
      auto path = pathFromString(stringValue);
      if (!path) {
        return toStatus(path.takeError());
      }
      file = std::move(*path);
    }
  }
  return QDMI_SUCCESS;
}

llvm::Expected<LoadedDeviceConfiguration>
loadDeviceConfiguration(const std::optional<std::string>& inlineJson,
                        const std::optional<std::filesystem::path>& file,
                        const std::string_view inlineEnvironment,
                        const std::string_view fileEnvironment,
                        const std::string_view bundledFilename,
                        const void* anchor) {
  if (inlineJson) {
    return LoadedDeviceConfiguration{
        .json = *inlineJson,
        .source = "inline session configuration",
    };
  }
  if (file) {
    return readFile(*file, false);
  }
  const auto environmentJson = llvm::sys::Process::GetEnv(inlineEnvironment);
  const auto environmentFile = llvm::sys::Process::GetEnv(fileEnvironment);
  if (environmentJson && environmentFile) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           llvm::formatv("Both {0} and {1} are set",
                                         inlineEnvironment, fileEnvironment)
                               .str());
  }
  if (environmentJson) {
    return LoadedDeviceConfiguration{
        .json = *environmentJson,
        .source = std::string(inlineEnvironment),
    };
  }
  if (environmentFile) {
    auto path = pathFromString(*environmentFile);
    if (!path) {
      return path.takeError();
    }
    return readFile(*path, false);
  }
  auto directory = moduleDirectory(anchor);
  if (!directory) {
    return directory.takeError();
  }
  return readFile(*directory / bundledFilename, true);
}
} // namespace qdmi::detail
