/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Common.cpp
/// Common definitions and utilities for working with QDMI in C++.

#include "qdmi/common/Common.hpp"

#include "qdmi/common/Diagnostics.hpp"

#include "qdmi/constants.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#ifdef _WIN32
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ConvertUTF.h"
#endif
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#ifdef _WIN32
#include <cstddef>
#endif
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace qdmi {
namespace detail {
llvm::Expected<std::string> pathToString(const std::filesystem::path& path) {
#ifdef _WIN32
  std::string utf8(path.native().size() * 3, '\0');
  const auto* source = reinterpret_cast<const llvm::UTF16*>(path.c_str());
  const auto* end = source + path.native().size();
  auto* output = reinterpret_cast<llvm::UTF8*>(utf8.data());
  // LLVM's Windows wrapper replaces invalid surrogates; the text wrapper strips
  // BOMs.
  if (llvm::ConvertUTF16toUTF8(&source, end, &output, output + utf8.size(),
                               llvm::strictConversion) != llvm::conversionOK) {
    return makeError(QDMI_ERROR_INVALIDARGUMENT,
                     "Invalid native filesystem path");
  }
  utf8.resize(
      static_cast<size_t>(reinterpret_cast<char*>(output) - utf8.data()));
  return utf8;
#else
  return path.native();
#endif
}

llvm::Expected<std::filesystem::path> pathFromString(std::string_view value) {
#ifdef _WIN32
  llvm::SmallVector<wchar_t, 256> native;
  if (auto error = llvm::sys::windows::UTF8ToUTF16(value, native)) {
    return makeError(QDMI_ERROR_INVALIDARGUMENT,
                     "Invalid UTF-8 filesystem path: " + error.message());
  }
  return std::filesystem::path(native.begin(), native.end());
#else
  return std::filesystem::path(value);
#endif
}
} // namespace detail

auto throwIfError(const int result, const std::string& msg) -> void {
  switch (const auto res = static_cast<QDMI_STATUS>(result)) {
  case QDMI_SUCCESS:
    break;
  case QDMI_WARN_GENERAL:
    qdmi::diagnostics::warn("{0}", msg);
    break;
  default:
    const auto message = llvm::formatv("{0}: {1}.", msg, toString(res)).str();
    switch (res) {
    case QDMI_ERROR_OUTOFMEM:
      throw std::bad_alloc();
    case QDMI_ERROR_OUTOFRANGE:
      throw std::out_of_range(message);
    case QDMI_ERROR_INVALIDARGUMENT:
      throw std::invalid_argument(message);
    case QDMI_ERROR_FATAL:
    case QDMI_ERROR_NOTIMPLEMENTED:
    case QDMI_ERROR_LIBNOTFOUND:
    case QDMI_ERROR_NOTFOUND:
    case QDMI_ERROR_PERMISSIONDENIED:
    case QDMI_ERROR_NOTSUPPORTED:
    case QDMI_ERROR_BADSTATE:
    case QDMI_ERROR_TIMEOUT:
      throw std::runtime_error(message);
    default:
      throw std::runtime_error("Unknown QDMI error code. " + message);
    }
  }
}

namespace {
class QDMIErrorCategory final : public std::error_category {
public:
  [[nodiscard]] const char* name() const noexcept override { return "qdmi"; }
  [[nodiscard]] std::string message(int status) const override {
    if (status >= QDMI_ERROR_TIMEOUT && status <= QDMI_ERROR_FATAL) {
      return toString(static_cast<QDMI_STATUS>(status));
    }
    return "Unknown QDMI error code " + std::to_string(status);
  }
  [[nodiscard]] std::error_condition
  default_error_condition(int status) const noexcept override {
    switch (status) {
    case QDMI_ERROR_INVALIDARGUMENT:
      return std::errc::invalid_argument;
    case QDMI_ERROR_OUTOFRANGE:
      return std::errc::result_out_of_range;
    case QDMI_ERROR_OUTOFMEM:
      return std::errc::not_enough_memory;
    default:
      return {status, *this};
    }
  }
};
} // namespace

llvm::Error makeError(const int status, const llvm::Twine& message) {
  static const QDMIErrorCategory CATEGORY;
  return llvm::createStringError(std::error_code(status, CATEGORY), message);
}

llvm::Error checkError(const int result, const llvm::Twine& message) {
  if (result == QDMI_SUCCESS) {
    return llvm::Error::success();
  }
  if (result == QDMI_WARN_GENERAL) {
    qdmi::diagnostics::warn("{0}", message.str());
    return llvm::Error::success();
  }
  if (result >= QDMI_ERROR_TIMEOUT && result <= QDMI_ERROR_FATAL) {
    return makeError(result, message + ": " +
                                 toString(static_cast<QDMI_STATUS>(result)) +
                                 ".");
  }
  return makeError(result, llvm::Twine("Unknown QDMI error code ") +
                               llvm::Twine(result) + ". " + message);
}

int toStatus(llvm::Error error) {
  if (!error) {
    return QDMI_SUCCESS;
  }
  int status = QDMI_ERROR_FATAL;
  llvm::visitErrors(error, [&](const llvm::ErrorInfoBase& info) {
    const auto code = info.convertToErrorCode();
    if (std::string_view(code.category().name()) == "qdmi") {
      status = code.value();
    }
  });
  qdmi::diagnostics::error("{0}", llvm::toString(std::move(error)));
  return status;
}
} // namespace qdmi

namespace qdmi::diagnostics::detail {
namespace {
[[nodiscard]] constexpr std::string_view
diagnosticLevelName(const DiagnosticLevel level) noexcept {
  switch (level) {
  case DiagnosticLevel::Info:
    return "info";
  case DiagnosticLevel::Warning:
    return "warning";
  case DiagnosticLevel::Error:
    return "error";
  }
  return "error";
}

/// Writes one diagnostic to standard error without allocating memory.
void writeDiagnostic(const DiagnosticLevel level,
                     const std::string_view message) noexcept {
#ifdef _WIN32
  _lock_file(stderr);
#else
  // NOLINTNEXTLINE(misc-include-cleaner)
  flockfile(stderr);
#endif
  const auto write = [](const std::string_view part) noexcept {
    if (!part.empty()) {
#ifdef _WIN32
      _fwrite_nolock(part.data(), sizeof(char), part.size(), stderr);
#else
      std::fwrite(part.data(), sizeof(char), part.size(), stderr);
#endif
    }
  };
  write("[mqt-core] [");
  write(diagnosticLevelName(level));
  write("] ");
  write(message);
#ifdef _WIN32
  _fputc_nolock('\n', stderr);
  _fflush_nolock(stderr);
  _unlock_file(stderr);
#else
  std::fputc('\n', stderr);
  std::fflush(stderr);
  // NOLINTNEXTLINE(misc-include-cleaner)
  funlockfile(stderr);
#endif
}

} // namespace

void emit(const DiagnosticLevel level,
          const llvm::formatv_object_base& message) noexcept {
  llvm::SmallString<256> buffer;
  llvm::raw_svector_ostream stream(buffer);
  stream << message;
  writeDiagnostic(level, std::string_view(buffer.data(), buffer.size()));
}
} // namespace qdmi::diagnostics::detail
