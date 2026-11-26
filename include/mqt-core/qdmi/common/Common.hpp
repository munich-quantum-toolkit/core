/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/// @file Common.hpp
/// Common definitions and utilities for working with QDMI in C++.

#pragma once

#include "qdmi/client.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>

namespace qdmi {
/// Custom scalars use their native C++ representation.
///
/// Byte spans borrow an exact, nonempty payload until the synchronous
/// submission call returns.
using CustomJobParameter =
    std::variant<std::string, bool, int, double, std::span<const std::byte>>;
/// Preserve a provider status in an owned LLVM error.
[[nodiscard]] llvm::Error makeError(int status, const llvm::Twine& message);

namespace detail {
/// Encode a native path for LLVM's UTF-8 Windows filesystem APIs.
[[nodiscard]] llvm::Expected<std::string>
pathToString(const std::filesystem::path& path);

/// Decode a path from LLVM's UTF-8 Windows filesystem representation.
[[nodiscard]] llvm::Expected<std::filesystem::path>
pathFromString(std::string_view value);
} // namespace detail

/// Return an error for failures and log provider warnings.
[[nodiscard]] llvm::Error checkError(int result, const llvm::Twine& message);

/// Log an owned error and return its original QDMI status at a C boundary.
int toStatus(llvm::Error error);

template <class Concrete> class Singleton {
protected:
  /// Protected constructor to enforce the singleton pattern.
  Singleton() = default;

public:
  // Delete move constructor and move assignment operator because of the
  // following reason:
  //
  // - Users access the singleton via get() which returns a reference
  // - Moving would invalidate the singleton's state
  // - Users never own the singleton instance
  Singleton(Singleton&&) = delete;
  Singleton& operator=(Singleton&&) = delete;
  // Delete copy constructor and assignment operator to enforce singleton.
  Singleton(const Singleton&) = delete;
  Singleton& operator=(const Singleton&) = delete;

  /// Virtual destructor for the Singleton base class.
  virtual ~Singleton() = default;

  /// Returns the singleton instance of the derived class.
  [[nodiscard]] static auto get() -> Concrete& {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    static auto* instance = new Concrete();
    // The instance is intentionally leaked to avoid static deinitialization
    // issues (cf. static (de)initialization order fiasco)
    return *instance;
  }
};

/// Report the size of `count` elements of type `T` in `sizeRet` and, when
/// `data` is set, let `fill` write them into the caller's buffer of `size`
/// bytes.
///
/// This is the provider side of the QDMI size-then-fill protocol. `fill`
/// writes typed elements, so `data` must be aligned for `T`; `writeValue` and
/// `writeList` copy bytes and accept any buffer.
template <typename T, std::invocable<std::span<T>> Fill>
  requires std::is_trivially_copy_constructible_v<T>
[[nodiscard]] QDMI_STATUS writeResult(const size_t count, const size_t size,
                                      void* data, size_t* sizeRet,
                                      Fill&& fill) {
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    return QDMI_ERROR_OUTOFMEM;
  }
  if (sizeRet != nullptr) {
    *sizeRet = count * sizeof(T);
  }
  if (data == nullptr) {
    return QDMI_SUCCESS;
  }
  if (size < count * sizeof(T)) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }
  std::forward<Fill>(fill)(std::span(static_cast<T*>(data), count));
  return QDMI_SUCCESS;
}

/// Write a list with the size-then-fill protocol.
template <std::ranges::contiguous_range Range>
  requires std::is_trivially_copy_constructible_v<
      std::ranges::range_value_t<Range>>
[[nodiscard]] QDMI_STATUS writeList(const Range& values, const size_t size,
                                    void* data, size_t* sizeRet) {
  using T = std::ranges::range_value_t<Range>;
  const auto count = std::ranges::size(values);
  if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
    return QDMI_ERROR_OUTOFMEM;
  }
  // Byte copies keep the caller's buffer free of alignment requirements.
  return writeResult<std::byte>(
      count * sizeof(T), size, data, sizeRet, [&](std::span<std::byte> output) {
        if (!output.empty()) {
          std::memcpy(output.data(), std::ranges::data(values), output.size());
        }
      });
}

/// Write one value with the size-then-fill protocol.
template <typename T>
  requires std::is_trivially_copy_constructible_v<T>
[[nodiscard]] QDMI_STATUS writeValue(const T& value, const size_t size,
                                     void* data, size_t* sizeRet) {
  return writeList(std::span(&value, 1), size, data, sizeRet);
}

/// Write a null-terminated string with the size-then-fill protocol.
[[nodiscard]] inline QDMI_STATUS writeString(const std::string_view value,
                                             const size_t size, void* data,
                                             size_t* sizeRet) {
  return writeResult<char>(value.size() + 1, size, data, sizeRet,
                           [&](std::span<char> output) {
                             std::ranges::copy(value, output.begin());
                             output.back() = '\0';
                           });
}

/// Return a value that the provider must support.
template <typename T>
[[nodiscard]] llvm::Expected<T>
requireSupported(llvm::Expected<std::optional<T>> value,
                 const llvm::Twine& message) {
  if (!value) {
    return value.takeError();
  }
  if (!*value) {
    return checkError(QDMI_ERROR_NOTSUPPORTED, message);
  }
  return std::move(**value);
}

/// Query the size in bytes of a variable-size value, or `std::nullopt` if the
/// provider does not support it.
///
/// This and `queryInto` are the client side of the protocol that
/// `writeResult` implements.
template <typename Query>
[[nodiscard]] llvm::Expected<std::optional<size_t>>
querySize(Query&& query, const llvm::Twine& message) {
  size_t size = 0;
  const auto status = query(0, nullptr, &size);
  if (status == QDMI_ERROR_NOTSUPPORTED) {
    return std::nullopt;
  }
  if (auto error = checkError(status, message)) {
    return error;
  }
  return size;
}

/// Read a variable-size value of exactly `buffer.size()` bytes.
template <typename Query>
[[nodiscard]] llvm::Error queryInto(Query&& query,
                                    const std::span<std::byte> buffer,
                                    const llvm::Twine& message) {
  // Providers that do not report the size on a read keep the expected one.
  size_t size = buffer.size();
  if (auto error = checkError(
          query(buffer.size(), buffer.empty() ? nullptr : buffer.data(), &size),
          message)) {
    return error;
  }
  if (size != buffer.size()) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           message + ": the value has " + llvm::Twine(size) +
                               " bytes, but the buffer holds " +
                               llvm::Twine(buffer.size()));
  }
  return llvm::Error::success();
}

/// Fetch a variable-size value into a new contiguous container, or return
/// `std::nullopt` if the provider does not support it.
///
/// The provider writes the bytes of `Buffer`'s elements, so they must be
/// trivially copy constructible. Containers with `resize_for_overwrite`, such
/// as `llvm::SmallVector`, skip initializing the elements.
template <typename Buffer, typename Query>
  requires std::is_trivially_copy_constructible_v<typename Buffer::value_type>
[[nodiscard]] llvm::Expected<std::optional<Buffer>>
queryBuffer(Query&& query, const llvm::Twine& sizeMessage,
            const llvm::Twine& message) {
  using Element = typename Buffer::value_type;
  auto size = querySize(query, sizeMessage);
  if (!size) {
    return size.takeError();
  }
  if (!*size) {
    return std::nullopt;
  }
  if (**size % sizeof(Element) != 0) {
    return qdmi::makeError(
        QDMI_ERROR_FATAL,
        sizeMessage + ": byte count is not a multiple of the element size");
  }
  std::optional<Buffer> buffer(std::in_place);
  if constexpr (requires { buffer->resize_for_overwrite(size_t{}); }) {
    buffer->resize_for_overwrite(**size / sizeof(Element));
  } else {
    buffer->resize(**size / sizeof(Element));
  }
  if (**size != 0) {
    if (auto error = queryInto(
            query, std::as_writable_bytes(std::span(*buffer)), message)) {
      return error;
    }
  }
  return buffer;
}

/// Fetch a variable-size value that the provider must support.
template <typename Buffer, typename Query>
[[nodiscard]] llvm::Expected<Buffer>
queryRequiredBuffer(Query&& query, const llvm::Twine& sizeMessage,
                    const llvm::Twine& message) {
  return requireSupported(
      queryBuffer<Buffer>(std::forward<Query>(query), sizeMessage, message),
      sizeMessage);
}

/// Returns the string representation of the given status code @p result.
constexpr auto toString(const QDMI_STATUS result) -> const char* {
  switch (result) {
  case QDMI_WARN_GENERAL:
    return "General warning";
  case QDMI_SUCCESS:
    return "Success";
  case QDMI_ERROR_FATAL:
    return "A fatal error";
  case QDMI_ERROR_OUTOFMEM:
    return "Out of memory";
  case QDMI_ERROR_NOTIMPLEMENTED:
    return "Not implemented";
  case QDMI_ERROR_LIBNOTFOUND:
    return "Library not found";
  case QDMI_ERROR_NOTFOUND:
    return "Element not found";
  case QDMI_ERROR_OUTOFRANGE:
    return "Out of range";
  case QDMI_ERROR_INVALIDARGUMENT:
    return "Invalid argument";
  case QDMI_ERROR_PERMISSIONDENIED:
    return "Permission denied";
  case QDMI_ERROR_NOTSUPPORTED:
    return "Not supported";
  case QDMI_ERROR_BADSTATE:
    return "Bad state";
  case QDMI_ERROR_TIMEOUT:
    return "Timeout";
  }
  llvm_unreachable("Invalid QDMI enum value");
}

/// Returns the string representation of the given session parameter @p param.
constexpr auto toString(const QDMI_Session_Parameter param) -> const char* {
  switch (param) {
  case QDMI_SESSION_PARAMETER_TOKEN:
    return "TOKEN";
  case QDMI_SESSION_PARAMETER_AUTHFILE:
    return "AUTH FILE";
  case QDMI_SESSION_PARAMETER_AUTHURL:
    return "AUTH URL";
  case QDMI_SESSION_PARAMETER_USERNAME:
    return "USERNAME";
  case QDMI_SESSION_PARAMETER_PASSWORD:
    return "PASSWORD";
  case QDMI_SESSION_PARAMETER_PROJECTID:
    return "PROJECT ID";
  case QDMI_SESSION_PARAMETER_MAX:
    return "MAX";
  case QDMI_SESSION_PARAMETER_CUSTOM1:
    return "CUSTOM1";
  case QDMI_SESSION_PARAMETER_CUSTOM2:
    return "CUSTOM2";
  case QDMI_SESSION_PARAMETER_CUSTOM3:
    return "CUSTOM3";
  case QDMI_SESSION_PARAMETER_CUSTOM4:
    return "CUSTOM4";
  case QDMI_SESSION_PARAMETER_CUSTOM5:
    return "CUSTOM5";
  }
  llvm_unreachable("Invalid QDMI enum value");
}

/// Returns the string representation of the given session property @p prop.
constexpr auto toString(const QDMI_Session_Property prop) -> const char* {
  switch (prop) {
  case QDMI_SESSION_PROPERTY_DEVICES:
    return "DEVICES";
  case QDMI_SESSION_PROPERTY_MAX:
    return "MAX";
  case QDMI_SESSION_PROPERTY_CUSTOM1:
    return "CUSTOM1";
  case QDMI_SESSION_PROPERTY_CUSTOM2:
    return "CUSTOM2";
  case QDMI_SESSION_PROPERTY_CUSTOM3:
    return "CUSTOM3";
  case QDMI_SESSION_PROPERTY_CUSTOM4:
    return "CUSTOM4";
  case QDMI_SESSION_PROPERTY_CUSTOM5:
    return "CUSTOM5";
  }
  llvm_unreachable("Invalid QDMI enum value");
}

/// Returns the string representation of the given device session parameter
/// @p param.
constexpr auto toString(const QDMI_Device_Session_Parameter param) -> const
    char* {
  switch (param) {
  case QDMI_DEVICE_SESSION_PARAMETER_BASEURL:
    return "BASE URL";
  case QDMI_DEVICE_SESSION_PARAMETER_TOKEN:
    return "TOKEN";
  case QDMI_DEVICE_SESSION_PARAMETER_AUTHFILE:
    return "AUTH FILE";
  case QDMI_DEVICE_SESSION_PARAMETER_AUTHURL:
    return "AUTH URL";
  case QDMI_DEVICE_SESSION_PARAMETER_USERNAME:
    return "USERNAME";
  case QDMI_DEVICE_SESSION_PARAMETER_PASSWORD:
    return "PASSWORD";
  case QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE:
    return "CHILD DEVICE";
  case QDMI_DEVICE_SESSION_PARAMETER_MAX:
    return "MAX";
  case QDMI_DEVICE_SESSION_PARAMETER_CUSTOM1:
    return "CUSTOM1";
  case QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2:
    return "CUSTOM2";
  case QDMI_DEVICE_SESSION_PARAMETER_CUSTOM3:
    return "CUSTOM3";
  case QDMI_DEVICE_SESSION_PARAMETER_CUSTOM4:
    return "CUSTOM4";
  case QDMI_DEVICE_SESSION_PARAMETER_CUSTOM5:
    return "CUSTOM5";
  }
  llvm_unreachable("Invalid QDMI enum value");
}

/// Returns the string representation of the given site property @p prop.
constexpr auto toString(const QDMI_Site_Property prop) -> const char* {
  switch (prop) {
  case QDMI_SITE_PROPERTY_INDEX:
    return "INDEX";
  case QDMI_SITE_PROPERTY_T1:
    return "T1";
  case QDMI_SITE_PROPERTY_T2:
    return "T2";
  case QDMI_SITE_PROPERTY_NAME:
    return "NAME";
  case QDMI_SITE_PROPERTY_XCOORDINATE:
    return "X COORDINATE";
  case QDMI_SITE_PROPERTY_YCOORDINATE:
    return "Y COORDINATE";
  case QDMI_SITE_PROPERTY_ZCOORDINATE:
    return "Z COORDINATE";
  case QDMI_SITE_PROPERTY_ISZONE:
    return "IS ZONE";
  case QDMI_SITE_PROPERTY_XEXTENT:
    return "X EXTENT";
  case QDMI_SITE_PROPERTY_YEXTENT:
    return "Y EXTENT";
  case QDMI_SITE_PROPERTY_ZEXTENT:
    return "Z EXTENT";
  case QDMI_SITE_PROPERTY_MODULEINDEX:
    return "MODULE INDEX";
  case QDMI_SITE_PROPERTY_SUBMODULEINDEX:
    return "SUBMODULE INDEX";
  case QDMI_SITE_PROPERTY_MAX:
    return "MAX";
  case QDMI_SITE_PROPERTY_CUSTOM1:
    return "CUSTOM1";
  case QDMI_SITE_PROPERTY_CUSTOM2:
    return "CUSTOM2";
  case QDMI_SITE_PROPERTY_CUSTOM3:
    return "CUSTOM3";
  case QDMI_SITE_PROPERTY_CUSTOM4:
    return "CUSTOM4";
  case QDMI_SITE_PROPERTY_CUSTOM5:
    return "CUSTOM5";
  }
  llvm_unreachable("Invalid QDMI enum value");
}

/// Returns the string representation of the given operation property @p prop.
constexpr auto toString(const QDMI_Operation_Property prop) -> const char* {
  switch (prop) {
  case QDMI_OPERATION_PROPERTY_NAME:
    return "NAME";
  case QDMI_OPERATION_PROPERTY_QUBITSNUM:
    return "QUBITS NUM";
  case QDMI_OPERATION_PROPERTY_PARAMETERSNUM:
    return "PARAMETERS NUM";
  case QDMI_OPERATION_PROPERTY_DURATION:
    return "DURATION";
  case QDMI_OPERATION_PROPERTY_FIDELITY:
    return "FIDELITY";
  case QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS:
    return "INTERACTION RADIUS";
  case QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS:
    return "BLOCKING RADIUS";
  case QDMI_OPERATION_PROPERTY_IDLINGFIDELITY:
    return "IDLING FIDELITY";
  case QDMI_OPERATION_PROPERTY_ISZONED:
    return "IS ZONED";
  case QDMI_OPERATION_PROPERTY_SITES:
    return "SITES";
  case QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED:
    return "MEAN SHUTTLING SPEED";
  case QDMI_OPERATION_PROPERTY_MAX:
    return "MAX";
  case QDMI_OPERATION_PROPERTY_CUSTOM1:
    return "CUSTOM1";
  case QDMI_OPERATION_PROPERTY_CUSTOM2:
    return "CUSTOM2";
  case QDMI_OPERATION_PROPERTY_CUSTOM3:
    return "CUSTOM3";
  case QDMI_OPERATION_PROPERTY_CUSTOM4:
    return "CUSTOM4";
  case QDMI_OPERATION_PROPERTY_CUSTOM5:
    return "CUSTOM5";
  }
  llvm_unreachable("Invalid QDMI enum value");
}

/// Returns the string representation of the given device property @p prop.
constexpr auto toString(const QDMI_Device_Property prop) -> const char* {
  switch (prop) {
  case QDMI_DEVICE_PROPERTY_ID:
    return "ID";
  case QDMI_DEVICE_PROPERTY_NAME:
    return "NAME";
  case QDMI_DEVICE_PROPERTY_VERSION:
    return "VERSION";
  case QDMI_DEVICE_PROPERTY_STATUS:
    return "STATUS";
  case QDMI_DEVICE_PROPERTY_LIBRARYVERSION:
    return "LIBRARY VERSION";
  case QDMI_DEVICE_PROPERTY_QUBITSNUM:
    return "QUBITS NUM";
  case QDMI_DEVICE_PROPERTY_SITES:
    return "SITES";
  case QDMI_DEVICE_PROPERTY_OPERATIONS:
    return "OPERATIONS";
  case QDMI_DEVICE_PROPERTY_COUPLINGMAP:
    return "COUPLING MAP";
  case QDMI_DEVICE_PROPERTY_LENGTHUNIT:
    return "LENGTH UNIT";
  case QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR:
    return "LENGTH SCALE FACTOR";
  case QDMI_DEVICE_PROPERTY_DURATIONUNIT:
    return "DURATION UNIT";
  case QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR:
    return "DURATION SCALE FACTOR";
  case QDMI_DEVICE_PROPERTY_MINATOMDISTANCE:
    return "MIN ATOM DISTANCE";
  case QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS:
    return "SUPPORTED PROGRAM FORMATS";
  case QDMI_DEVICE_PROPERTY_CHILDDEVICES:
    return "CHILD DEVICES";
  case QDMI_DEVICE_PROPERTY_QUEUELENGTH:
    return "QUEUE LENGTH";
  case QDMI_DEVICE_PROPERTY_MAX:
    return "MAX";
  case QDMI_DEVICE_PROPERTY_CUSTOM1:
    return "CUSTOM1";
  case QDMI_DEVICE_PROPERTY_CUSTOM2:
    return "CUSTOM2";
  case QDMI_DEVICE_PROPERTY_CUSTOM3:
    return "CUSTOM3";
  case QDMI_DEVICE_PROPERTY_CUSTOM4:
    return "CUSTOM4";
  case QDMI_DEVICE_PROPERTY_CUSTOM5:
    return "CUSTOM5";
  }
  llvm_unreachable("Invalid QDMI enum value");
}

} // namespace qdmi
