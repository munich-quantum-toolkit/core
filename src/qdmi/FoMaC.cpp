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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <qdmi/client.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fomac {
auto checkError(int result, const std::string& msg) -> void {
  if (result != QDMI_SUCCESS) {
    std::stringstream ss;
    ss << msg << ": ";
    switch (result) {
    case QDMI_WARN_GENERAL:
      ss << "A general warning occurred.";
      spdlog::warn(ss.str());
      break;
    case QDMI_ERROR_OUTOFMEM:
      throw std::bad_alloc();
    case QDMI_ERROR_OUTOFRANGE:
      ss << "Out of range.";
      throw std::out_of_range(ss.str());
    case QDMI_ERROR_INVALIDARGUMENT:
      ss << "Invalid argument.";
      throw std::invalid_argument(ss.str());
    default: { /* all errors that result in a runtime exception */
      switch (result) {
      case QDMI_ERROR_FATAL:
        ss << "Fatal error.";
        break;
      case QDMI_ERROR_NOTIMPLEMENTED:
        ss << "Not implemented.";
        break;
      case QDMI_ERROR_LIBNOTFOUND:
        ss << "Library not found.";
        break;
      case QDMI_ERROR_NOTFOUND:
        ss << "Not found.";
        break;
      case QDMI_ERROR_PERMISSIONDENIED:
        ss << "Permission denied.";
        break;
      case QDMI_ERROR_NOTSUPPORTED:
        ss << "Not supported.";
        break;
      case QDMI_ERROR_BADSTATE:
        ss << "Bad state.";
        break;
      case QDMI_ERROR_TIMEOUT:
        ss << "Timeout.";
        break;
      default:
        ss << "Unknown error.";
        break;
      }
      throw std::runtime_error(ss.str());
    }
    }
  }
}
auto Site::getIndex() const -> size_t {
  return queryProperty<size_t>(QDMI_SITE_PROPERTY_INDEX);
}
auto Site::getT1() const -> uint64_t {
  return queryProperty<uint64_t>(QDMI_SITE_PROPERTY_T1);
}
auto Site::getT2() const -> uint64_t {
  return queryProperty<uint64_t>(QDMI_SITE_PROPERTY_T2);
}
auto Site::getName() const -> std::string {
  return queryProperty<std::string>(QDMI_SITE_PROPERTY_NAME);
}
auto Site::getXCoordinate() const -> int64_t {
  return queryProperty<int64_t>(QDMI_SITE_PROPERTY_XCOORDINATE);
}
auto Site::getYCoordinate() const -> int64_t {
  return queryProperty<int64_t>(QDMI_SITE_PROPERTY_YCOORDINATE);
}
auto Site::getZCoordinate() const -> int64_t {
  return queryProperty<int64_t>(QDMI_SITE_PROPERTY_ZCOORDINATE);
}
auto Site::isZone() const -> bool {
  return queryProperty<bool>(QDMI_SITE_PROPERTY_ISZONE);
}
auto Site::getXExtent() const -> uint64_t {
  return queryProperty<uint64_t>(QDMI_SITE_PROPERTY_XEXTENT);
}
auto Site::getYExtent() const -> uint64_t {
  return queryProperty<uint64_t>(QDMI_SITE_PROPERTY_YEXTENT);
}
auto Site::getZExtent() const -> uint64_t {
  return queryProperty<uint64_t>(QDMI_SITE_PROPERTY_ZEXTENT);
}
auto Site::getModuleIndex() const -> uint64_t {
  return queryProperty<uint64_t>(QDMI_SITE_PROPERTY_MODULEINDEX);
}
auto Site::getSubmoduleIndex() const -> uint64_t {
  return queryProperty<uint64_t>(QDMI_SITE_PROPERTY_SUBMODULEINDEX);
}
auto Operation::getName(const std::vector<Site>& sites,
                        const std::vector<double>& params) -> std::string {
  return queryProperty<std::string>(QDMI_OPERATION_PROPERTY_NAME, sites,
                                    params);
}
auto Operation::getQubitsNum(const std::vector<Site>& sites,
                             const std::vector<double>& params) -> size_t {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_QUBITSNUM, sites,
                               params);
}
auto Operation::getParametersNum(const std::vector<Site>& sites,
                                 const std::vector<double>& params) -> size_t {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sites,
                               params);
}
auto Operation::getDuration(const std::vector<Site>& sites,
                            const std::vector<double>& params) -> uint64_t {
  return queryProperty<uint64_t>(QDMI_OPERATION_PROPERTY_DURATION, sites,
                                 params);
}
auto Operation::getFidelity(const std::vector<Site>& sites,
                            const std::vector<double>& params) -> double {
  return queryProperty<double>(QDMI_OPERATION_PROPERTY_FIDELITY, sites, params);
}
auto Operation::getInteractionRadius(const std::vector<Site>& sites,
                                     const std::vector<double>& params)
    -> uint64_t {
  return queryProperty<uint64_t>(QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS,
                                 sites, params);
}
auto Operation::getBlockingRadius(const std::vector<Site>& sites,
                                  const std::vector<double>& params)
    -> uint64_t {
  return queryProperty<uint64_t>(QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS, sites,
                                 params);
}
auto Operation::getIdlingFidelity(const std::vector<Site>& sites,
                                  const std::vector<double>& params) -> double {
  return queryProperty<double>(QDMI_OPERATION_PROPERTY_IDLINGFIDELITY, sites,
                               params);
}
auto Operation::isZoned(const std::vector<Site>& sites,
                        const std::vector<double>& params) -> bool {
  return queryProperty<bool>(QDMI_OPERATION_PROPERTY_ISZONED, sites, params);
}
auto Operation::getSites(const std::vector<Site>& sites,
                         const std::vector<double>& params)
    -> std::vector<Site> {
  const auto& qdmiSites = queryProperty<std::vector<QDMI_Site>>(
      QDMI_OPERATION_PROPERTY_SITES, sites, params);
  std::vector<Site> returnedSites;
  returnedSites.reserve(qdmiSites.size());
  std::ranges::transform(
      qdmiSites, std::back_inserter(returnedSites),
      [this](const QDMI_Site& site) -> Site { return {device_, site}; });
  return returnedSites;
}
auto Operation::getMeanShuttlingSpeed(const std::vector<Site>& sites,
                                      const std::vector<double>& params)
    -> uint64_t {
  return queryProperty<uint64_t>(QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED,
                                 sites, params);
}
} // namespace fomac
