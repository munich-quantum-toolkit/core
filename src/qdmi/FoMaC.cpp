/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/FoMaC.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <qdmi/client.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fomac {
#define THROW_IF_ERROR(stmt, msg)                                              \
  if (const auto result = stmt; result != QDMI_SUCCESS) {                      \
    std::stringstream ss;                                                      \
    ss << "[" __FILE__ ":" << __LINE__ << "] " << (msg) << ":";                \
    switch (result) {                                                          \
    case QDMI_WARN_GENERAL:                                                    \
      SPDLOG_WARN("A general warning occurred.");                              \
      break;                                                                   \
    case QDMI_ERROR_OUTOFMEM:                                                  \
      throw std::bad_alloc();                                                  \
    case QDMI_ERROR_OUTOFRANGE:                                                \
      ss << "Out of range.";                                                   \
      throw std::out_of_range(ss.str());                                       \
    case QDMI_ERROR_INVALIDARGUMENT:                                           \
      ss << "Invalid argument.";                                               \
      throw std::invalid_argument(ss.str());                                   \
    default: { /* all errors that result in a runtime exception */             \
      switch (result) {                                                        \
      case QDMI_ERROR_FATAL:                                                   \
        ss << "Fatal error.";                                                  \
        break;                                                                 \
      case QDMI_ERROR_NOTIMPLEMENTED:                                          \
        ss << "Not implemented.";                                              \
        break;                                                                 \
      case QDMI_ERROR_LIBNOTFOUND:                                             \
        ss << "Library not found.";                                            \
        break;                                                                 \
      case QDMI_ERROR_NOTFOUND:                                                \
        ss << "Element not found.";                                            \
        break;                                                                 \
      case QDMI_ERROR_PERMISSIONDENIED:                                        \
        ss << "Permission denied.";                                            \
        break;                                                                 \
      case QDMI_ERROR_NOTSUPPORTED:                                            \
        ss << "Not supported.";                                                \
        break;                                                                 \
      case QDMI_ERROR_BADSTATE:                                                \
        ss << "Bad state.";                                                    \
        break;                                                                 \
      case QDMI_ERROR_TIMEOUT:                                                 \
        ss << "Timeout.";                                                      \
        break;                                                                 \
      default:                                                                 \
        ss << "Unknown error (" << result << ").";                             \
        break;                                                                 \
      }                                                                        \
      throw std::runtime_error(ss.str());                                      \
      break;                                                                   \
    }                                                                          \
    }                                                                          \
  }

#define QUERY_SINGLE_VALUE_PROPERTY(prop, type, value)                         \
  type(value) = 0;                                                             \
  THROW_IF_ERROR(QDMI_device_query_site_property(                              \
                     device_, site_, (prop), sizeof(type), &(value), nullptr), \
                 "Querying " #prop)

#define QUERY_STRING_PROPERTY(prop, value)                                     \
  size_t size = 0;                                                             \
  THROW_IF_ERROR(QDMI_device_query_site_property(device_, site_, (prop), 0,    \
                                                 nullptr, &size),              \
                 "Querying " #prop);                                           \
  std::string(value)(size - 1, '\0');                                          \
  THROW_IF_ERROR(QDMI_device_query_site_property(device_, site_, (prop), size, \
                                                 (value).data(), nullptr),     \
                 "Querying " #prop);

auto Site::getIndex() const -> size_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_INDEX, size_t, index);
  return index;
}
auto Site::getT1() const -> uint64_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T1, uint64_t, t1);
  return t1;
}
auto Site::getT2() const -> uint64_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_T2, uint64_t, t2);
  return t2;
}
auto Site::getName() const -> std::string {
  QUERY_STRING_PROPERTY(QDMI_SITE_PROPERTY_NAME, name);
  return name;
}
auto Site::getXCoordinate() const -> int64_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_XCOORDINATE, int64_t,
                              xCoordinate);
  return xCoordinate;
}
auto Site::getYCoordinate() const -> int64_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_YCOORDINATE, int64_t,
                              yCoordinate);
  return yCoordinate;
}
auto Site::getZCoordinate() const -> int64_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_ZCOORDINATE, int64_t,
                              zCoordinate);
  return zCoordinate;
}
auto Site::isZone() const -> bool {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_ISZONE, bool, zone);
  return zone;
}
auto Site::getXExtent() const -> uint64_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_XEXTENT, uint64_t, xExtent);
  return xExtent;
}
auto Site::getYExtent() const -> uint64_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_YEXTENT, uint64_t, yExtent);
  return yExtent;
}
auto Site::getZExtent() const -> uint64_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_ZEXTENT, uint64_t, zExtent);
  return zExtent;
}
auto Site::getModuleIndex() const -> size_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_MODULEINDEX, size_t,
                              moduleIndex);
  return moduleIndex;
}
auto Site::getSubmoduleIndex() const -> size_t {
  QUERY_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_SUBMODULEINDEX, size_t,
                              submoduleIndex);
  return submoduleIndex;
}
} // namespace fomac
