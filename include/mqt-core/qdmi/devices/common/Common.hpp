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
/// @brief Common definitions and utilities for working with QDMI in C++.
/// @note This header will be upstreamed to the QDMI core library in the future.

#pragma once

#include <qdmi/client.h>
#include <qdmi/constants.h>
#include <string>

namespace qdmi {
/**
 * @brief Function used to mark unreachable code
 * @details Uses compiler specific extensions if possible. Even if no extension
 * is used, undefined behavior is still raised by an empty function body and the
 * noreturn attribute.
 */
[[noreturn]] inline void unreachable() {
#ifdef __GNUC__ // GCC, Clang, ICC
  __builtin_unreachable();
#elif defined(_MSC_VER) // MSVC
  __assume(false);
#endif
}

// NOLINTBEGIN(bugprone-macro-parentheses)
#define ADD_SINGLE_VALUE_PROPERTY(prop_name, prop_type, prop_value, prop,      \
                                  size, value, size_ret)                       \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < sizeof(prop_type)) {                                      \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        *static_cast<prop_type*>(value) = prop_value;                          \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = sizeof(prop_type);                                         \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

// STRNCPY wrapper: strncpy_s on Windows (auto null-terminates),
// strncpy on other platforms (requires manual null-termination - see usage).
#ifdef _WIN32
#define STRNCPY(dest, src, size)                                               \
  strncpy_s(static_cast<char*>(dest), size, src, size);
#else
#define STRNCPY(dest, src, size) strncpy(static_cast<char*>(dest), src, size);
#endif

#define ADD_STRING_PROPERTY(prop_name, prop_value, prop, size, value,          \
                            size_ret)                                          \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < strlen(prop_value) + 1) {                                 \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        STRNCPY(value, prop_value, size);                                      \
        /* Ensure null-termination: strncpy doesn't guarantee it on non-Win */ \
        /* NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic) */  \
        static_cast<char*>(value)[size - 1] = '\0';                            \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = strlen(prop_value) + 1;                                    \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

#define ADD_LIST_PROPERTY(prop_name, prop_type, prop_values, prop, size,       \
                          value, size_ret)                                     \
  {                                                                            \
    if ((prop) == (prop_name)) {                                               \
      if ((value) != nullptr) {                                                \
        if ((size) < (prop_values).size() * sizeof(prop_type)) {               \
          return QDMI_ERROR_INVALIDARGUMENT;                                   \
        }                                                                      \
        memcpy(static_cast<void*>(value),                                      \
               static_cast<const void*>((prop_values).data()),                 \
               (prop_values).size() * sizeof(prop_type));                      \
      }                                                                        \
      if ((size_ret) != nullptr) {                                             \
        *size_ret = (prop_values).size() * sizeof(prop_type);                  \
      }                                                                        \
      return QDMI_SUCCESS;                                                     \
    }                                                                          \
  }

#define IS_INVALID_ARGUMENT(prop, prefix)                                      \
  ((prop) >= prefix##_MAX && (prop) != prefix##_CUSTOM1 &&                     \
   (prop) != prefix##_CUSTOM2 && (prop) != prefix##_CUSTOM3 &&                 \
   (prop) != prefix##_CUSTOM4 && (prop) != prefix##_CUSTOM5)
// NOLINTEND(bugprone-macro-parentheses)

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
  unreachable();
}

/**
 * @brief Throws an exception if the result indicates an error.
 * @param result The result of a QDMI operation
 * @param msg The error message to include in the exception
 * @throws std::bad_alloc if the result is QDMI_ERROR_OUTOFMEM
 * @throws std::out_of_range if the result is QDMI_ERROR_OUTOFRANGE
 * @throws std::invalid_argument if the result is QDMI_ERROR_INVALIDARGUMENT
 * @throws std::runtime_error for all other error results
 */
auto throwIfError(int result, const std::string& msg) -> void;

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
  unreachable();
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
  unreachable();
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
  unreachable();
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
  unreachable();
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
  unreachable();
}

/// Returns the string representation of the given device property @p prop.
constexpr auto toString(const QDMI_Device_Property prop) -> const char* {
  switch (prop) {
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
  case QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION:
    return "NEEDS CALIBRATION";
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
  case QDMI_DEVICE_PROPERTY_PULSESUPPORT:
    return "PULSE SUPPORT";
  case QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS:
    return "SUPPORTED PROGRAM FORMATS";
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
  unreachable();
}

} // namespace qdmi
