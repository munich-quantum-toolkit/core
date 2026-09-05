/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file Client.hpp
 * @brief QDMI C++ device-management interface.
 */

#pragma once

#include "qdmi/common/Common.hpp"
#include "qdmi/types.h"

#include <qdmi/client.h>

#include <algorithm>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace qdmi {
using CustomJobParameter = std::variant<std::string, bool, int, double>;

/**
 * @brief Identifies one of QDMI's implementation-defined custom slots.
 * @details The same selector is used for custom device, site, operation, and
 * job properties as well as custom job results.
 */
enum class CustomProperty : std::uint8_t {
  Custom1 = 1,
  Custom2 = 2,
  Custom3 = 3,
  Custom4 = 4,
  Custom5 = 5,
};

/**
 * @brief Concept for supported custom property value types.
 * @details Raw bytes provide a lossless fallback for implementation-defined
 * types that cannot be represented by one of the scalar alternatives.
 */
template <typename T>
concept custom_property_value =
    std::same_as<T, std::string> || std::same_as<T, bool> ||
    std::same_as<T, int> || std::same_as<T, double> ||
    std::same_as<T, std::vector<std::byte>>;

namespace detail {
struct ClientApi {
  decltype(&::QDMI_driver_get_client_abi_version) driverGetClientAbiVersion{};
  decltype(&::QDMI_session_alloc) sessionAlloc{};
  decltype(&::QDMI_session_init) sessionInit{};
  decltype(&::QDMI_session_free) sessionFree{};
  decltype(&::QDMI_session_set_parameter) sessionSetParameter{};
  decltype(&::QDMI_session_query_session_property) sessionQueryProperty{};
  decltype(&::QDMI_device_create_job) deviceCreateJob{};
  decltype(&::QDMI_session_retrieve_job_by_id) sessionRetrieveJobById{};
  decltype(&::QDMI_job_free) jobFree{};
  decltype(&::QDMI_job_set_parameter) jobSetParameter{};
  decltype(&::QDMI_job_query_property) jobQueryProperty{};
  decltype(&::QDMI_job_submit) jobSubmit{};
  decltype(&::QDMI_job_cancel) jobCancel{};
  decltype(&::QDMI_job_check) jobCheck{};
  decltype(&::QDMI_job_wait) jobWait{};
  decltype(&::QDMI_job_get_results) jobGetResults{};
  decltype(&::QDMI_device_query_device_property) deviceQueryProperty{};
  decltype(&::QDMI_device_query_site_property) deviceQuerySiteProperty{};
  decltype(&::QDMI_device_query_operation_property)
      deviceQueryOperationProperty{};
};

struct ClientSession {
  ClientSession(std::shared_ptr<const ClientApi> selectedApi,
                QDMI_Session selectedSession)
      : api(std::move(selectedApi)), handle(selectedSession) {}
  ~ClientSession();

  ClientSession(const ClientSession&) = delete;
  ClientSession& operator=(const ClientSession&) = delete;

  std::shared_ptr<const ClientApi> api;
  QDMI_Session handle;
};

struct JobDeleter {
  void operator()(QDMI_Job_impl_d* job) const;
  std::shared_ptr<ClientSession> session;
};

[[nodiscard]] inline std::string
decodeText(std::string value, const std::string_view description) {
  if (value.empty() || value.back() != '\0') {
    throw std::invalid_argument(std::string(description) +
                                " is not null-terminated");
  }
  if (value.find('\0') != value.size() - 1U) {
    throw std::invalid_argument(std::string(description) +
                                " contains an embedded null byte");
  }
  value.pop_back();
  return value;
}

[[nodiscard]] inline std::string
decodeText(const std::span<const std::byte> value,
           const std::string_view description) {
  if (value.empty()) {
    throw std::invalid_argument(std::string(description) +
                                " is not null-terminated");
  }
  return decodeText(
      std::string{reinterpret_cast<const char*>(value.data()), value.size()},
      description);
}

[[nodiscard]] inline std::optional<size_t>
queuePositionFromResult(const int result, const size_t queuePosition) {
  if (result == QDMI_ERROR_NOTSUPPORTED || result == QDMI_ERROR_BADSTATE) {
    return std::nullopt;
  }
  qdmi::throwIfError(result, "Querying job queue position");
  return queuePosition;
}

[[nodiscard]] inline std::vector<std::string>
parseShots(const std::string_view shots, const size_t numShots) {
  if (numShots == 0) {
    if (!shots.empty()) {
      throw std::runtime_error("Number of shots mismatch");
    }
    return {};
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
    throw std::runtime_error("Number of shots mismatch");
  }
  return parsed;
}

template <custom_property_value T, typename Query>
[[nodiscard]] std::optional<T>
queryCustomValue(Query query, const std::string_view description) {
  size_t size = 0;
  const auto sizeResult = query(0, nullptr, &size);
  if (sizeResult == QDMI_ERROR_NOTSUPPORTED) {
    return std::nullopt;
  }
  qdmi::throwIfError(sizeResult,
                     "Querying " + std::string(description) + " size");

  std::vector<std::byte> bytes(size);
  if (size != 0) {
    qdmi::throwIfError(query(size, bytes.data(), nullptr),
                       "Querying " + std::string(description));
  }

  if constexpr (std::same_as<T, std::vector<std::byte>>) {
    return bytes;
  } else if constexpr (std::same_as<T, std::string>) {
    if (bytes.empty() || bytes.back() != std::byte{0}) {
      throw std::invalid_argument("Cannot decode " + std::string(description) +
                                  " as a null-terminated string");
    }
    return std::string(reinterpret_cast<const char*>(bytes.data()),
                       bytes.size() - 1);
  } else {
    if (bytes.size() != sizeof(T)) {
      throw std::invalid_argument("Cannot decode " + std::string(description) +
                                  ": expected " + std::to_string(sizeof(T)) +
                                  " bytes, but the device reported " +
                                  std::to_string(bytes.size()));
    }
    T value{};
    std::memcpy(&value, bytes.data(), sizeof(T));
    return value;
  }
}

template <typename Element>
void validateArraySize(const size_t size, const std::string_view description) {
  if (size % sizeof(Element) != 0U) {
    throw std::invalid_argument(
        "Cannot decode " + std::string(description) + ": the device reported " +
        std::to_string(size) + " bytes, which is not a multiple of " +
        std::to_string(sizeof(Element)));
  }
}

template <typename Handle, typename Query>
[[nodiscard]] std::optional<std::vector<Handle>>
queryHandleArray(Query query, const std::string_view description) {
  size_t size = 0;
  const auto sizeResult = query(0, nullptr, &size);
  if (sizeResult == QDMI_ERROR_NOTSUPPORTED) {
    return std::nullopt;
  }
  qdmi::throwIfError(sizeResult,
                     "Querying " + std::string(description) + " size");
  validateArraySize<Handle>(size, description);

  std::vector<Handle> handles(size / sizeof(Handle));
  if (size != 0) {
    qdmi::throwIfError(query(size, static_cast<void*>(handles.data()), nullptr),
                       "Querying " + std::string(description));
  }
  return handles;
}

[[nodiscard]] constexpr QDMI_Device_Property
toDeviceProperty(const CustomProperty property) {
  switch (property) {
  case CustomProperty::Custom1:
    return QDMI_DEVICE_PROPERTY_CUSTOM1;
  case CustomProperty::Custom2:
    return QDMI_DEVICE_PROPERTY_CUSTOM2;
  case CustomProperty::Custom3:
    return QDMI_DEVICE_PROPERTY_CUSTOM3;
  case CustomProperty::Custom4:
    return QDMI_DEVICE_PROPERTY_CUSTOM4;
  case CustomProperty::Custom5:
    return QDMI_DEVICE_PROPERTY_CUSTOM5;
  }
  throw std::invalid_argument("Invalid custom property selector");
}

[[nodiscard]] constexpr QDMI_Site_Property
toSiteProperty(const CustomProperty property) {
  switch (property) {
  case CustomProperty::Custom1:
    return QDMI_SITE_PROPERTY_CUSTOM1;
  case CustomProperty::Custom2:
    return QDMI_SITE_PROPERTY_CUSTOM2;
  case CustomProperty::Custom3:
    return QDMI_SITE_PROPERTY_CUSTOM3;
  case CustomProperty::Custom4:
    return QDMI_SITE_PROPERTY_CUSTOM4;
  case CustomProperty::Custom5:
    return QDMI_SITE_PROPERTY_CUSTOM5;
  }
  throw std::invalid_argument("Invalid custom property selector");
}

[[nodiscard]] constexpr QDMI_Operation_Property
toOperationProperty(const CustomProperty property) {
  switch (property) {
  case CustomProperty::Custom1:
    return QDMI_OPERATION_PROPERTY_CUSTOM1;
  case CustomProperty::Custom2:
    return QDMI_OPERATION_PROPERTY_CUSTOM2;
  case CustomProperty::Custom3:
    return QDMI_OPERATION_PROPERTY_CUSTOM3;
  case CustomProperty::Custom4:
    return QDMI_OPERATION_PROPERTY_CUSTOM4;
  case CustomProperty::Custom5:
    return QDMI_OPERATION_PROPERTY_CUSTOM5;
  }
  throw std::invalid_argument("Invalid custom property selector");
}

[[nodiscard]] constexpr QDMI_Job_Property
toJobProperty(const CustomProperty property) {
  switch (property) {
  case CustomProperty::Custom1:
    return QDMI_JOB_PROPERTY_CUSTOM1;
  case CustomProperty::Custom2:
    return QDMI_JOB_PROPERTY_CUSTOM2;
  case CustomProperty::Custom3:
    return QDMI_JOB_PROPERTY_CUSTOM3;
  case CustomProperty::Custom4:
    return QDMI_JOB_PROPERTY_CUSTOM4;
  case CustomProperty::Custom5:
    return QDMI_JOB_PROPERTY_CUSTOM5;
  }
  throw std::invalid_argument("Invalid custom property selector");
}

[[nodiscard]] constexpr QDMI_Job_Result
toJobResult(const CustomProperty property) {
  switch (property) {
  case CustomProperty::Custom1:
    return QDMI_JOB_RESULT_CUSTOM1;
  case CustomProperty::Custom2:
    return QDMI_JOB_RESULT_CUSTOM2;
  case CustomProperty::Custom3:
    return QDMI_JOB_RESULT_CUSTOM3;
  case CustomProperty::Custom4:
    return QDMI_JOB_RESULT_CUSTOM4;
  case CustomProperty::Custom5:
    return QDMI_JOB_RESULT_CUSTOM5;
  }
  throw std::invalid_argument("Invalid custom property selector");
}
} // namespace detail

/**
 * @brief Returns whether a program format carries a binary payload.
 * @details `QDMI_PROGRAM_FORMAT_QIRBASEMODULE`,
 * `QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE`, and `QDMI_PROGRAM_FORMAT_QPY` hold
 * bitcode or another serialized object. Such a payload can contain a null byte
 * and is not text, so it must be submitted as exact bytes. The string overload
 * of `Device::submitJob` rejects these formats.
 * @param format The program format to classify.
 * @return True if the format requires exact-byte submission.
 */
[[nodiscard]] constexpr bool
isBinaryProgramFormat(const QDMI_Program_Format format) noexcept {
  return format == QDMI_PROGRAM_FORMAT_QIRBASEMODULE ||
         format == QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE ||
         format == QDMI_PROGRAM_FORMAT_QPY;
}

/**
 * @brief Concept for ranges that are contiguous in memory and can be
 * constructed with a size.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept size_constructible_contiguous_range =
    std::ranges::contiguous_range<T> && std::constructible_from<T, size_t> &&
    requires { typename T::value_type; } && requires(T t) {
      { t.data() } -> std::same_as<typename T::value_type*>;
    };
/**
 * @brief Concept for types that are either integral, floating point, bool,
 * std::string, or QDMI_Device_Status.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept value_or_string =
    std::integral<T> || std::floating_point<T> || std::same_as<T, bool> ||
    std::same_as<T, std::string> || std::same_as<T, QDMI_Device_Status>;

/**
 * @brief Concept for types that are either value_or_string or
 * size_constructible_contiguous_range.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept value_or_string_or_vector =
    value_or_string<T> || size_constructible_contiguous_range<T>;

/**
 * @brief Concept for types that are std::optional of value_or_string.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept is_optional = requires { typename T::value_type; } &&
                      std::same_as<T, std::optional<typename T::value_type>>;

/**
 * @brief Concept for types that are either std::string or std::optional of
 * std::string.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 */
template <typename T>
concept string_or_optional_string =
    std::same_as<T, std::string> ||
    (is_optional<T> && std::same_as<typename T::value_type, std::string>);

/// @see remove_optional_t
/// The name follows the standard-library type-trait convention.
/// NOLINTNEXTLINE(readability-identifier-naming)
template <typename T> struct remove_optional {
  using type = T;
};

/// @see remove_optional_t
template <typename U> struct remove_optional<std::optional<U>> {
  using type = U;
};

/**
 * @brief Helper type to strip std::optional from a type if it is present.
 * @details This is useful for template metaprogramming when you want to work
 * with the underlying type of optional without caring about its optionality.
 * @tparam T The type to strip optional from.
 */
template <typename T> using remove_optional_t = remove_optional<T>::type;

/**
 * @brief Concept for types that are either size_constructible_contiguous_range
 * or std::optional of size_constructible_contiguous_range.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 * @see Operation::queryProperty
 */
template <typename T>
concept maybe_optional_size_constructible_contiguous_range =
    size_constructible_contiguous_range<remove_optional_t<T>>;

/**
 * @brief Concept for types that are either value_or_string or std::optional of
 * value_or_string.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 * @see Site::queryProperty
 */
template <typename T>
concept maybe_optional_value_or_string = value_or_string<remove_optional_t<T>>;

/**
 * @brief Concept for types that are either value_or_string_or_vector or
 * std::optional of value_or_string_or_vector.
 * @details This concept is used to constrain the template parameter of the
 * `queryProperty` method.
 * @tparam T The type to check.
 * @see Operation::queryProperty
 */
template <typename T>
concept maybe_optional_value_or_string_or_vector =
    value_or_string_or_vector<remove_optional_t<T>>;

/**
 * @brief Configuration structure for session authentication parameters.
 * @details All parameters are optional. Only set the parameters needed for
 * your authentication method. Parameters are validated when the session is
 * constructed.
 */
struct SessionConfig {
  /// QDMI Client driver library. Uses the environment or packaged driver when
  /// omitted.
  std::optional<std::filesystem::path> driverPath;
  /// Authentication token
  std::optional<std::string> token;
  /// Path to file containing authentication information
  std::optional<std::filesystem::path> authFile;
  /// URL to authentication server
  std::optional<std::string> authUrl;
  /// Username for authentication
  std::optional<std::string> username;
  /// Password for authentication
  std::optional<std::string> password;
  /// Project ID for session
  std::optional<std::string> projectId;
  /// Custom configuration parameter 1
  std::optional<std::string> custom1;
  /// Custom configuration parameter 2
  std::optional<std::string> custom2;
  /// Custom configuration parameter 3
  std::optional<std::string> custom3;
  /// Custom configuration parameter 4
  std::optional<std::string> custom4;
  /// Custom configuration parameter 5
  std::optional<std::string> custom5;
};

class Job;
class Site;
class Device;
class Operation;

/**
 * @brief Class representing the Session library.
 * @details This class provides methods to query available devices and
 * manage the QDMI session.
 * @see QDMI_Session
 */
class Session {
public:
  /**
   * @brief Opens a Client-visible QDMI device in a fresh session.
   * @param id Stable device ID.
   * @param config Client driver and authentication configuration.
   * @return A device wrapper that retains the fresh session.
   */
  [[nodiscard]] static Device openDevice(std::string_view id,
                                         const SessionConfig& config = {});

  /**
   * @brief Constructs a new QDMI Session with optional authentication.
   * @param config Optional session configuration containing authentication
   * parameters. If not provided, uses default (no authentication).
   * @details Creates, allocates, and initializes a new QDMI session.
   */
  explicit Session(const SessionConfig& config = {});

  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  Session(Session&&) noexcept = default;
  Session& operator=(Session&&) noexcept = default;

  /// @see QDMI_SESSION_PROPERTY_DEVICES
  [[nodiscard]] std::vector<Device> getDevices();

private:
  [[nodiscard]] const detail::ClientApi& api() const { return *session_->api; }

  /// Query a session property.
  template <size_constructible_contiguous_range T>
  [[nodiscard]] T queryProperty(const QDMI_Session_Property prop) const {
    using StrippedValueType = remove_optional_t<T>::value_type;

    size_t size = 0;
    qdmi::throwIfError(session_->api->sessionQueryProperty(
                           session_->handle, prop, 0, nullptr, &size),
                       std::string("Querying size ") + qdmi::toString(prop));
    detail::validateArraySize<StrippedValueType>(size, qdmi::toString(prop));
    remove_optional_t<T> value(size / sizeof(StrippedValueType));
    qdmi::throwIfError(session_->api->sessionQueryProperty(
                           session_->handle, prop, size,
                           static_cast<void*>(value.data()), nullptr),
                       std::string("Querying ") + qdmi::toString(prop));
    return value;
  }

  std::shared_ptr<detail::ClientSession> session_;
};

static_assert(!std::is_copy_constructible<Session>());
static_assert(!std::is_copy_assignable<Session>());
static_assert(std::is_move_constructible<Session>());
static_assert(std::is_move_assignable<Session>());

/**
 * @brief Class representing a quantum device.
 * @details
 * This class provides methods to query properties of the device,
 * its sites, and its operations.
 *
 * The class can only be constructed by Session instances.
 *
 * @see QDMI_Device
 */
class Device {
public:
  // NOLINTNEXTLINE(google-explicit-constructor, *-explicit-conversions)
  operator QDMI_Device() const { return device_; }

  /// @see QDMI_DEVICE_PROPERTY_ID
  [[nodiscard]] std::string getId() const;

  /// @see QDMI_DEVICE_PROPERTY_NAME
  [[nodiscard]] std::string getName() const;

  /// @see QDMI_DEVICE_PROPERTY_VERSION
  [[nodiscard]] std::string getVersion() const;

  /// @see QDMI_DEVICE_PROPERTY_STATUS
  [[nodiscard]] QDMI_Device_Status getStatus() const;

  /// @see QDMI_DEVICE_PROPERTY_LIBRARYVERSION
  [[nodiscard]] std::string getLibraryVersion() const;

  /// @see QDMI_DEVICE_PROPERTY_QUBITSNUM
  [[nodiscard]] size_t getQubitsNum() const;

  /// @see QDMI_DEVICE_PROPERTY_SITES
  [[nodiscard]] std::vector<Site> getSites() const;

  /**
   * @brief Returns the list of regular sites (without zone sites) available
   * on the device.
   * @details Filters all sites and only returns regular sites, i.e., where
   * `isZone()` yields `false`. These represent actual potential physical
   * qubit locations on the device lattice.
   * @returns vector of regular sites
   * @see QDMI_DEVICE_PROPERTY_SITES
   */
  [[nodiscard]] std::vector<Site> getRegularSites() const;

  /**
   * @brief Returns the list of zone sites (without regular sites) available
   * on the device.
   * @details Filters all sites and only returns zone sites, i.e., where
   * `isZone()` yields `true`. These represent a zone, i.e., an extent where
   * zoned operations can be performed, not individual qubit locations.
   * @returns a vector of zone sites
   * @see QDMI_DEVICE_PROPERTY_SITES
   */
  [[nodiscard]] std::vector<Site> getZones() const;

  /// @see QDMI_DEVICE_PROPERTY_OPERATIONS
  [[nodiscard]] std::vector<Operation> getOperations() const;

  /// @see QDMI_DEVICE_PROPERTY_COUPLINGMAP
  [[nodiscard]] std::optional<std::vector<std::pair<Site, Site>>>
  getCouplingMap() const;

  /// @see QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION
  [[nodiscard]] std::optional<size_t> getNeedsCalibration() const;

  /// @see QDMI_DEVICE_PROPERTY_QUEUELENGTH
  [[nodiscard]] std::optional<size_t> getQueueLength() const;

  /// @see QDMI_DEVICE_PROPERTY_LENGTHUNIT
  [[nodiscard]] std::optional<std::string> getLengthUnit() const;

  /// @see QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR
  [[nodiscard]] std::optional<double> getLengthScaleFactor() const;

  /// @see QDMI_DEVICE_PROPERTY_DURATIONUNIT
  [[nodiscard]] std::optional<std::string> getDurationUnit() const;

  /// @see QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR
  [[nodiscard]] std::optional<double> getDurationScaleFactor() const;

  /// @see QDMI_DEVICE_PROPERTY_MINATOMDISTANCE
  [[nodiscard]] std::optional<uint64_t> getMinAtomDistance() const;

  /// @see QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS
  [[nodiscard]] std::vector<QDMI_Program_Format>
  getSupportedProgramFormats() const;

  /**
   * @brief Returns the direct child devices managed by this device.
   * @return The child devices, or an empty vector if child devices are not
   * supported.
   * @see QDMI_DEVICE_PROPERTY_CHILDDEVICES
   */
  [[nodiscard]] std::vector<Device> getChildDevices() const;

  /**
   * @brief Queries an implementation-defined custom device property.
   * @tparam T Expected value type. Use `std::vector<std::byte>` to retrieve the
   * raw value without interpretation.
   * @param property Custom property slot to query.
   * @return The decoded value, or `std::nullopt` if the slot is unsupported.
   * @throws std::invalid_argument If the returned bytes do not match `T`.
   */
  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  queryCustomProperty(const CustomProperty property) const {
    const auto qdmiProperty = detail::toDeviceProperty(property);
    return detail::queryCustomValue<T>(
        [this, qdmiProperty](const size_t size, void* value, size_t* sizeRet) {
          return session_->api->deviceQueryProperty(device_, qdmiProperty, size,
                                                    value, sizeRet);
        },
        "custom device property " +
            std::to_string(static_cast<unsigned>(property)));
  }

  /**
   * @brief Queries a custom device property containing operation handles.
   * @param property Custom property slot to query.
   * @return Normal QDMI operation wrappers, or `std::nullopt` if the slot is
   * unsupported. A supported empty list is returned as an engaged optional.
   * @throws std::invalid_argument If the returned byte count is not a multiple
   * of `sizeof(QDMI_Operation)`.
   */
  [[nodiscard]] std::optional<std::vector<Operation>>
  queryCustomOperations(CustomProperty property) const;

  /**
   * @brief Submits a textual program.
   * @details The terminating null byte required by QDMI text formats is
   * included in the submitted payload.
   * @throws std::invalid_argument If the format requires binary submission,
   * names a batch job, or names a calibration run.
   * @see QDMI_job_submit
   */
  [[nodiscard]] Job submitJob(
      const std::string& program, QDMI_Program_Format format, size_t numShots,
      const std::optional<CustomJobParameter>& custom1 = std::nullopt,
      const std::optional<CustomJobParameter>& custom2 = std::nullopt,
      const std::optional<CustomJobParameter>& custom3 = std::nullopt,
      const std::optional<CustomJobParameter>& custom4 = std::nullopt,
      const std::optional<CustomJobParameter>& custom5 = std::nullopt) const;

  /**
   * @brief Submits a textual program without setting a shot count.
   * @details Repetition semantics are left to the submitted program and device.
   * @see QDMI_job_submit
   */
  [[nodiscard]] Job submitJob(
      const std::string& program, QDMI_Program_Format format,
      const std::optional<CustomJobParameter>& custom1 = std::nullopt,
      const std::optional<CustomJobParameter>& custom2 = std::nullopt,
      const std::optional<CustomJobParameter>& custom3 = std::nullopt,
      const std::optional<CustomJobParameter>& custom4 = std::nullopt,
      const std::optional<CustomJobParameter>& custom5 = std::nullopt) const;

  /**
   * @brief Submits a binary program.
   * @details The bytes are submitted exactly as provided without appending a
   * null byte.
   * @throws std::invalid_argument If the format names a batch job or a
   * calibration run.
   * @see QDMI_job_submit
   */
  [[nodiscard]] Job submitJob(
      std::span<const std::byte> program, QDMI_Program_Format format,
      size_t numShots,
      const std::optional<CustomJobParameter>& custom1 = std::nullopt,
      const std::optional<CustomJobParameter>& custom2 = std::nullopt,
      const std::optional<CustomJobParameter>& custom3 = std::nullopt,
      const std::optional<CustomJobParameter>& custom4 = std::nullopt,
      const std::optional<CustomJobParameter>& custom5 = std::nullopt) const;

  /**
   * @brief Submits a binary program without setting a shot count.
   * @details Repetition semantics are left to the submitted program and device.
   * @see QDMI_job_submit
   */
  [[nodiscard]] Job submitJob(
      std::span<const std::byte> program, QDMI_Program_Format format,
      const std::optional<CustomJobParameter>& custom1 = std::nullopt,
      const std::optional<CustomJobParameter>& custom2 = std::nullopt,
      const std::optional<CustomJobParameter>& custom3 = std::nullopt,
      const std::optional<CustomJobParameter>& custom4 = std::nullopt,
      const std::optional<CustomJobParameter>& custom5 = std::nullopt) const;

  /**
   * @brief Triggers a calibration run.
   * @details A device that reports a nonzero
   * `QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION` is asked to calibrate by submitting
   * a job in the `QDMI_PROGRAM_FORMAT_CALIBRATION` format. QDMI does not
   * require a program for such a job, so the payload is optional; when it is
   * present, the device defines what it means, which is usually a
   * configuration for the run. A calibration run executes no circuit, so no
   * shot count is set.
   * @param program The calibration payload. An empty span or `std::nullopt`
   * means that the job has no payload.
   * @see QDMI_job_submit
   */
  [[nodiscard]] Job submitCalibrationJob(
      std::optional<std::span<const std::byte>> program = std::nullopt,
      const std::optional<CustomJobParameter>& custom1 = std::nullopt,
      const std::optional<CustomJobParameter>& custom2 = std::nullopt,
      const std::optional<CustomJobParameter>& custom3 = std::nullopt,
      const std::optional<CustomJobParameter>& custom4 = std::nullopt,
      const std::optional<CustomJobParameter>& custom5 = std::nullopt) const;

  /**
   * @brief Triggers a calibration run with a text payload.
   * @details The terminating null byte required by QDMI text formats is
   * included in the submitted payload.
   * @param program The calibration payload.
   * @see QDMI_job_submit
   */
  [[nodiscard]] Job submitCalibrationJob(
      const std::string& program,
      const std::optional<CustomJobParameter>& custom1 = std::nullopt,
      const std::optional<CustomJobParameter>& custom2 = std::nullopt,
      const std::optional<CustomJobParameter>& custom3 = std::nullopt,
      const std::optional<CustomJobParameter>& custom4 = std::nullopt,
      const std::optional<CustomJobParameter>& custom5 = std::nullopt) const;

  /**
   * @brief Retrieves an existing job by its device-provided ID.
   * @details Opening a job does not submit, clone, or modify the remote job.
   * The returned handle can be used to query its state and retrieve results.
   * @param jobId The nonempty opaque ID returned by @ref Job::getId.
   * @throws std::runtime_error If the driver or device cannot retrieve the job.
   * @see QDMI_session_retrieve_job_by_id
   */
  [[nodiscard]] Job retrieveJobById(std::string_view jobId) const;

  auto operator<=>(const Device&) const noexcept = default;

private:
  [[nodiscard]] const detail::ClientApi& api() const { return *session_->api; }

  /**
   * @brief Constructs a Device object from a QDMI_Device handle.
   * @param device The QDMI_Device handle to wrap.
   * @param session The Client session that owns the handle.
   */
  Device(QDMI_Device device, std::shared_ptr<detail::ClientSession> session)
      : device_(device), session_(std::move(session)) {}

  /// Wrap operation handles while retaining their owning device session.
  [[nodiscard]] std::vector<Operation>
  wrapOperations(std::span<const QDMI_Operation> operations) const;

  /// Query a device property.
  template <maybe_optional_value_or_string_or_vector T>
  [[nodiscard]] T queryProperty(const QDMI_Device_Property prop) const {
    std::string msg = "Querying ";
    msg += qdmi::toString(prop);

    if constexpr (string_or_optional_string<T>) {
      size_t size = 0;
      auto result =
          session_->api->deviceQueryProperty(device_, prop, 0, nullptr, &size);

      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }

      qdmi::throwIfError(result, msg);
      std::string value(size, '\0');
      result = session_->api->deviceQueryProperty(device_, prop, size,
                                                  value.data(), nullptr);
      qdmi::throwIfError(result, msg);
      return detail::decodeText(std::move(value), msg);
    } else if constexpr (maybe_optional_size_constructible_contiguous_range<
                             T>) {
      size_t size = 0;
      auto result =
          session_->api->deviceQueryProperty(device_, prop, 0, nullptr, &size);

      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }

      qdmi::throwIfError(result, msg);
      detail::validateArraySize<typename remove_optional_t<T>::value_type>(
          size, qdmi::toString(prop));
      remove_optional_t<T> value(
          size / sizeof(typename remove_optional_t<T>::value_type));
      result = session_->api->deviceQueryProperty(
          device_, prop, size, static_cast<void*>(value.data()), nullptr);
      qdmi::throwIfError(result, msg);
      return value;
    } else {
      remove_optional_t<T> value{};
      const auto result = session_->api->deviceQueryProperty(
          device_, prop, sizeof(remove_optional_t<T>), &value, nullptr);

      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }

      qdmi::throwIfError(result, msg);
      return value;
    }
  }

  [[nodiscard]] Job
  submitJobImpl(QDMI_Program_Format format,
                std::optional<std::span<const std::byte>> program,
                std::optional<size_t> numShots,
                const std::optional<CustomJobParameter>& custom1,
                const std::optional<CustomJobParameter>& custom2,
                const std::optional<CustomJobParameter>& custom3,
                const std::optional<CustomJobParameter>& custom4,
                const std::optional<CustomJobParameter>& custom5) const;

  void setCustomJobParam(QDMI_Job job, QDMI_Job_Parameter param,
                         const CustomJobParameter& value) const;

  /// @brief The underlying device pointer.
  QDMI_Device device_{};
  std::shared_ptr<detail::ClientSession> session_;

  friend class Session;
};

/**
 * @brief Class representing a submitted job.
 * @details
 * This class provides methods to query job status and retrieve
 * results.
 *
 * The class can only be constructed by Device instances.
 *
 * @see QDMI_Job
 */
class Job {
public:
  Job(Job&&) noexcept = default;
  Job& operator=(Job&&) noexcept = default;

  // NOLINTNEXTLINE(misc-explicit-constructor, *-explicit-conversions)
  operator QDMI_Job() const { return job_.get(); }

  /// @see QDMI_job_check
  [[nodiscard]] QDMI_Job_Status check() const;

  /**
   * @brief @see QDMI_job_wait
   * @param timeout The maximum time to wait in seconds. 0 (default) means
   * wait indefinitely.
   * @return true if the job completed successfully, false if it timed out
   */
  [[nodiscard]] bool wait(size_t timeout = 0) const;

  /// @see QDMI_job_cancel
  void cancel() const;

  /// Get the job ID
  [[nodiscard]] std::string getId() const;

  /// Get the program format
  [[nodiscard]] QDMI_Program_Format getProgramFormat() const;

  /**
   * @brief Gets a textual program without its terminating null byte.
   * @throws std::invalid_argument If the format is not textual or the device
   * does not return a null-terminated payload.
   */
  [[nodiscard]] std::string getProgram() const;

  /**
   * @brief Gets the submitted program bytes exactly as returned by the device.
   */
  [[nodiscard]] std::vector<std::byte> getProgramBytes() const;

  /// Get the number of shots
  [[nodiscard]] size_t getNumShots() const;

  /**
   * @brief Gets the current number of jobs ahead of this job in its queue.
   * @return The queue position, or `std::nullopt` if it is unavailable or not
   * applicable in the job's current state.
   * @throws std::runtime_error If the provider status refresh or property query
   * fails for another reason.
   * @see QDMI_JOB_PROPERTY_QUEUEPOSITION
   */
  [[nodiscard]] std::optional<size_t> getQueuePosition() const;

  /**
   * @brief Queries an implementation-defined custom job property.
   * @tparam T Expected value type. Use `std::vector<std::byte>` to retrieve the
   * raw value without interpretation.
   * @param property Custom property slot to query.
   * @return The decoded value, or `std::nullopt` if the slot is unsupported.
   * @throws std::invalid_argument If the returned bytes do not match `T`.
   */
  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  queryCustomProperty(const CustomProperty property) const {
    const auto qdmiProperty = detail::toJobProperty(property);
    return detail::queryCustomValue<T>(
        [this, qdmiProperty](const size_t size, void* value, size_t* sizeRet) {
          return job_.get_deleter().session->api->jobQueryProperty(
              job_.get(), qdmiProperty, size, value, sizeRet);
        },
        "custom job property " +
            std::to_string(static_cast<unsigned>(property)));
  }

  /**
   * @brief Retrieves an implementation-defined custom job result.
   * @tparam T Expected value type. Use `std::vector<std::byte>` to retrieve the
   * raw value without interpretation.
   * @param property Custom result slot to query.
   * @return The decoded value, or `std::nullopt` if the slot is unsupported.
   * @throws std::invalid_argument If the returned bytes do not match `T`.
   */
  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  getCustomResult(const CustomProperty property) const {
    const auto qdmiResult = detail::toJobResult(property);
    return detail::queryCustomValue<T>(
        [this, qdmiResult](const size_t size, void* value, size_t* sizeRet) {
          return job_.get_deleter().session->api->jobGetResults(
              job_.get(), qdmiResult, size, value, sizeRet);
        },
        "custom job result " + std::to_string(static_cast<unsigned>(property)));
  }

  /**
   * @brief Returns the measurement shots as a vector of bitstrings.
   * @see QDMI_JOB_RESULT_SHOTS
   */
  [[nodiscard]] std::vector<std::string> getShots() const;

  /**
   * @brief Returns a map of measurement outcomes to their respective counts.
   * @see QDMI_JOB_RESULT_HIST_KEYS
   * @see QDMI_JOB_RESULT_HIST_VALUES
   */
  [[nodiscard]] std::map<std::string, size_t> getCounts() const;

  /**
   * @brief Returns the dense state vector as a vector of complex numbers.
   * @see QDMI_JOB_RESULT_STATEVECTOR_DENSE
   */
  [[nodiscard]] std::vector<std::complex<double>> getDenseStateVector() const;

  /**
   * @brief Returns the dense probabilities as a vector of doubles.
   * @see QDMI_JOB_RESULT_PROBABILITIES_DENSE
   */
  [[nodiscard]] std::vector<double> getDenseProbabilities() const;

  /**
   * @brief Returns the sparse state vector as a map of bitstrings to complex
   * amplitudes.
   * @see QDMI_JOB_RESULT_STATEVECTOR_SPARSE_KEYS
   * @see QDMI_JOB_RESULT_STATEVECTOR_SPARSE_VALUES
   */
  [[nodiscard]] std::map<std::string, std::complex<double>>
  getSparseStateVector() const;

  /**
   * @brief Returns the sparse probabilities as a map of bitstrings to
   * probabilities.
   * @see QDMI_JOB_RESULT_PROBABILITIES_SPARSE_KEYS
   * @see QDMI_JOB_RESULT_PROBABILITIES_SPARSE_VALUES
   */
  [[nodiscard]] std::map<std::string, double> getSparseProbabilities() const;

  auto operator<=>(const Job&) const noexcept = default;

private:
  [[nodiscard]] const detail::ClientApi& api() const {
    return *job_.get_deleter().session->api;
  }

  /**
   * @brief Constructs a Job object from a QDMI_Job handle.
   * @param job The QDMI_Job handle to wrap.
   * @param session The Client session that owns the handle.
   */
  Job(QDMI_Job job, std::shared_ptr<detail::ClientSession> session)
      : job_(job, detail::JobDeleter{std::move(session)}) {}

  std::unique_ptr<QDMI_Job_impl_d, detail::JobDeleter> job_;

  friend class Device;
};

static_assert(!std::is_copy_constructible<Job>());
static_assert(!std::is_copy_assignable<Job>());
static_assert(std::is_move_constructible<Job>());
static_assert(std::is_move_assignable<Job>());

/**
 * @brief Class representing a site (qubit) on the device.
 * @details
 * This class provides methods to query properties of the site.
 *
 * The class can only be constructed by Device and Operation instances.
 *
 * @see QDMI_Site
 */
class Site {
public:
  // NOLINTNEXTLINE(misc-explicit-constructor, *-explicit-conversions)
  operator QDMI_Site() const { return site_; }

  /// @see QDMI_SITE_PROPERTY_INDEX
  [[nodiscard]] size_t getIndex() const;

  /// @see QDMI_SITE_PROPERTY_T1
  [[nodiscard]] std::optional<uint64_t> getT1() const;

  /// @see QDMI_SITE_PROPERTY_T2
  [[nodiscard]] std::optional<uint64_t> getT2() const;

  /// @see QDMI_SITE_PROPERTY_NAME
  [[nodiscard]] std::optional<std::string> getName() const;

  /// @see QDMI_SITE_PROPERTY_XCOORDINATE
  [[nodiscard]] std::optional<int64_t> getXCoordinate() const;

  /// @see QDMI_SITE_PROPERTY_YCOORDINATE
  [[nodiscard]] std::optional<int64_t> getYCoordinate() const;

  /// @see QDMI_SITE_PROPERTY_ZCOORDINATE
  [[nodiscard]] std::optional<int64_t> getZCoordinate() const;

  /// @see QDMI_SITE_PROPERTY_ISZONE
  [[nodiscard]] bool isZone() const;

  /// @see QDMI_SITE_PROPERTY_XEXTENT
  [[nodiscard]] std::optional<uint64_t> getXExtent() const;

  /// @see QDMI_SITE_PROPERTY_YEXTENT
  [[nodiscard]] std::optional<uint64_t> getYExtent() const;

  /// @see QDMI_SITE_PROPERTY_ZEXTENT
  [[nodiscard]] std::optional<uint64_t> getZExtent() const;

  /// @see QDMI_SITE_PROPERTY_MODULEINDEX
  [[nodiscard]] std::optional<uint64_t> getModuleIndex() const;

  /// @see QDMI_SITE_PROPERTY_SUBMODULEINDEX
  [[nodiscard]] std::optional<uint64_t> getSubmoduleIndex() const;

  /**
   * @brief Queries an implementation-defined custom site property.
   * @tparam T Expected value type. Use `std::vector<std::byte>` to retrieve the
   * raw value without interpretation.
   * @param property Custom property slot to query.
   * @return The decoded value, or `std::nullopt` if the slot is unsupported.
   * @throws std::invalid_argument If the returned bytes do not match `T`.
   */
  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  queryCustomProperty(const CustomProperty property) const {
    const auto qdmiProperty = detail::toSiteProperty(property);
    return detail::queryCustomValue<T>(
        [this, qdmiProperty](const size_t size, void* value, size_t* sizeRet) {
          return session_->api->deviceQuerySiteProperty(
              device_, site_, qdmiProperty, size, value, sizeRet);
        },
        "custom site property " +
            std::to_string(static_cast<unsigned>(property)));
  }

  auto operator<=>(const Site&) const noexcept = default;

private:
  [[nodiscard]] const detail::ClientApi& api() const { return *session_->api; }

  /**
   * @brief Constructs a Site object from a QDMI_Site handle.
   * @param device The QDMI device handle that owns the site.
   * @param session The Client session that owns the handle.
   * @param site The QDMI_Site handle to wrap.
   */
  Site(QDMI_Device device, std::shared_ptr<detail::ClientSession> session,
       QDMI_Site site)
      : device_(device), session_(std::move(session)), site_(site) {}

  /// Query a site property.
  template <maybe_optional_value_or_string T>
  [[nodiscard]] T queryProperty(const QDMI_Site_Property prop) const {
    if constexpr (string_or_optional_string<T>) {
      size_t size = 0;
      const auto result = session_->api->deviceQuerySiteProperty(
          device_, site_, prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result,
                         std::string("Querying size") + qdmi::toString(prop));
      std::string value(size, '\0');
      qdmi::throwIfError(session_->api->deviceQuerySiteProperty(
                             device_, site_, prop, size, value.data(), nullptr),
                         std::string("Querying ") + qdmi::toString(prop));
      return detail::decodeText(std::move(value), qdmi::toString(prop));
    } else {
      remove_optional_t<T> value{};
      const auto result = session_->api->deviceQuerySiteProperty(
          device_, site_, prop, sizeof(remove_optional_t<T>), &value, nullptr);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result,
                         std::string("Querying ") + qdmi::toString(prop));
      return value;
    }
  }

  /// @brief The QDMI device handle that owns the site.
  QDMI_Device device_{};
  std::shared_ptr<detail::ClientSession> session_;

  /// @brief The underlying QDMI_Site object.
  QDMI_Site site_;

  friend class Device;
  friend class Operation;
};

/**
 * @brief Class representing an operation (gate) supported by the device.
 * @details
 * This class provides methods to query properties of the
 * operation.
 *
 * The class can only be constructed by Device instances.
 *
 * @see QDMI_Operation
 */
class Operation {
public:
  // NOLINTNEXTLINE(misc-explicit-constructor, *-explicit-conversions)
  operator QDMI_Operation() const { return operation_; }

  /// @see QDMI_OPERATION_PROPERTY_NAME
  [[nodiscard]] std::string
  getName(const std::vector<Site>& sites = {},
          const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_QUBITSNUM
  [[nodiscard]] std::optional<size_t>
  getQubitsNum(const std::vector<Site>& sites = {},
               const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_PARAMETERSNUM
  [[nodiscard]] size_t
  getParametersNum(const std::vector<Site>& sites = {},
                   const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_DURATION
  [[nodiscard]] std::optional<uint64_t>
  getDuration(const std::vector<Site>& sites = {},
              const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_FIDELITY
  [[nodiscard]] std::optional<double>
  getFidelity(const std::vector<Site>& sites = {},
              const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS
  [[nodiscard]] std::optional<uint64_t>
  getInteractionRadius(const std::vector<Site>& sites = {},
                       const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS
  [[nodiscard]] std::optional<uint64_t>
  getBlockingRadius(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_IDLINGFIDELITY
  [[nodiscard]] std::optional<double>
  getIdlingFidelity(const std::vector<Site>& sites = {},
                    const std::vector<double>& params = {}) const;

  /// @see QDMI_OPERATION_PROPERTY_ISZONED
  [[nodiscard]] bool isZoned() const;

  /// @see QDMI_OPERATION_PROPERTY_SITES
  [[nodiscard]] std::optional<std::vector<Site>> getSites() const;

  /**
   * @brief Returns the list of site pairs the local 2-qubit operation can
   * be performed on.
   * @details For local 2-qubit operations, this function interprets the
   * returned list of sites by QDMI as site pairs according to the QDMI
   * specification. Hence, this function facilitates easier iteration over
   * supported site pairs.
   * @return Optional vector of site pairs if this is a local 2-qubit
   * operation, std::nullopt otherwise.
   * @see QDMI_OPERATION_PROPERTY_SITES
   */
  [[nodiscard]] std::optional<std::vector<std::pair<Site, Site>>>
  getSitePairs() const;

  /// @see QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED
  [[nodiscard]] std::optional<uint64_t>
  getMeanShuttlingSpeed(const std::vector<Site>& sites = {},
                        const std::vector<double>& params = {}) const;

  /**
   * @brief Queries an implementation-defined custom operation property.
   * @tparam T Expected value type. Use `std::vector<std::byte>` to retrieve the
   * raw value without interpretation.
   * @param property Custom property slot to query.
   * @param sites Sites for context-dependent operation properties.
   * @param params Parameters for context-dependent operation properties.
   * @return The decoded value, or `std::nullopt` if the slot is unsupported.
   * @throws std::invalid_argument If the returned bytes do not match `T`.
   */
  template <custom_property_value T>
  [[nodiscard]] std::optional<T>
  queryCustomProperty(const CustomProperty property,
                      const std::vector<Site>& sites = {},
                      const std::vector<double>& params = {}) const {
    const auto qdmiProperty = detail::toOperationProperty(property);
    std::vector<QDMI_Site> qdmiSites;
    qdmiSites.reserve(sites.size());
    std::ranges::transform(sites, std::back_inserter(qdmiSites),
                           [](const Site& site) -> QDMI_Site { return site; });
    return detail::queryCustomValue<T>(
        [this, qdmiProperty, &qdmiSites,
         &params](const size_t size, void* value, size_t* sizeRet) {
          return session_->api->deviceQueryOperationProperty(
              device_, operation_, qdmiSites.size(), qdmiSites.data(),
              params.size(), params.data(), qdmiProperty, size, value, sizeRet);
        },
        "custom operation property " +
            std::to_string(static_cast<unsigned>(property)));
  }

  auto operator<=>(const Operation&) const noexcept = default;

private:
  [[nodiscard]] const detail::ClientApi& api() const { return *session_->api; }

  /**
   * @brief Constructs an Operation object from a QDMI_Operation handle.
   * @param device The QDMI device handle that owns the operation.
   * @param session The Client session that owns the handle.
   * @param operation The QDMI_Operation handle to wrap.
   */
  Operation(QDMI_Device device, std::shared_ptr<detail::ClientSession> session,
            QDMI_Operation operation)
      : device_(device), session_(std::move(session)), operation_(operation) {}

  /// Query an operation property.
  template <maybe_optional_value_or_string_or_vector T>
  [[nodiscard]] T queryProperty(const QDMI_Operation_Property prop,
                                const std::vector<Site>& sites,
                                const std::vector<double>& params) const {
    std::string msg = "Querying ";
    msg += qdmi::toString(prop);
    std::vector<QDMI_Site> qdmiSites;
    qdmiSites.reserve(sites.size());
    std::ranges::transform(sites, std::back_inserter(qdmiSites),
                           [](const Site& site) -> QDMI_Site { return site; });
    if constexpr (string_or_optional_string<T>) {
      size_t size = 0;
      auto result = session_->api->deviceQueryOperationProperty(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result, msg);
      std::string value(size, '\0');
      result = session_->api->deviceQueryOperationProperty(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, size, value.data(), nullptr);
      qdmi::throwIfError(result, msg);
      return detail::decodeText(std::move(value), msg);
    } else if constexpr (maybe_optional_size_constructible_contiguous_range<
                             T>) {
      size_t size = 0;
      auto result = session_->api->deviceQueryOperationProperty(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, 0, nullptr, &size);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result, msg);
      detail::validateArraySize<typename remove_optional_t<T>::value_type>(
          size, qdmi::toString(prop));
      remove_optional_t<T> value(
          size / sizeof(typename remove_optional_t<T>::value_type));
      result = session_->api->deviceQueryOperationProperty(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, size, static_cast<void*>(value.data()), nullptr);
      qdmi::throwIfError(result, msg);
      return value;
    } else {
      remove_optional_t<T> value{};
      const auto result = session_->api->deviceQueryOperationProperty(
          device_, operation_, sites.size(), qdmiSites.data(), params.size(),
          params.data(), prop, sizeof(remove_optional_t<T>), &value, nullptr);
      if constexpr (is_optional<T>) {
        if (result == QDMI_ERROR_NOTSUPPORTED) {
          return std::nullopt;
        }
      }
      qdmi::throwIfError(result, msg);
      return value;
    }
  }

  /// @brief The QDMI device handle that owns the operation.
  QDMI_Device device_{};
  std::shared_ptr<detail::ClientSession> session_;

  /// @brief The underlying QDMI_Operation object.
  QDMI_Operation operation_;

  friend class Device;
};
} // namespace qdmi
