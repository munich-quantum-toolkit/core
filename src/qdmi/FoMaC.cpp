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

#include "qdmi/Driver.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <qdmi/client.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
                        const std::vector<double>& params) const
    -> std::string {
  return queryProperty<std::string>(QDMI_OPERATION_PROPERTY_NAME, sites,
                                    params);
}
auto Operation::getQubitsNum(const std::vector<Site>& sites,
                             const std::vector<double>& params) const
    -> size_t {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_QUBITSNUM, sites,
                               params);
}
auto Operation::getParametersNum(const std::vector<Site>& sites,
                                 const std::vector<double>& params) const
    -> size_t {
  return queryProperty<size_t>(QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sites,
                               params);
}
auto Operation::getDuration(const std::vector<Site>& sites,
                            const std::vector<double>& params) const
    -> uint64_t {
  return queryProperty<uint64_t>(QDMI_OPERATION_PROPERTY_DURATION, sites,
                                 params);
}
auto Operation::getFidelity(const std::vector<Site>& sites,
                            const std::vector<double>& params) const -> double {
  return queryProperty<double>(QDMI_OPERATION_PROPERTY_FIDELITY, sites, params);
}
auto Operation::getInteractionRadius(const std::vector<Site>& sites,
                                     const std::vector<double>& params) const
    -> uint64_t {
  return queryProperty<uint64_t>(QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS,
                                 sites, params);
}
auto Operation::getBlockingRadius(const std::vector<Site>& sites,
                                  const std::vector<double>& params) const
    -> uint64_t {
  return queryProperty<uint64_t>(QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS, sites,
                                 params);
}
auto Operation::getIdlingFidelity(const std::vector<Site>& sites,
                                  const std::vector<double>& params) const
    -> double {
  return queryProperty<double>(QDMI_OPERATION_PROPERTY_IDLINGFIDELITY, sites,
                               params);
}
auto Operation::isZoned(const std::vector<Site>& sites,
                        const std::vector<double>& params) const -> bool {
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
                                      const std::vector<double>& params) const
    -> uint64_t {
  return queryProperty<uint64_t>(QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED,
                                 sites, params);
}
auto Device::getName() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_NAME);
}
auto Device::getVersion() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_VERSION);
}
auto Device::getStatus() const -> QDMI_Device_Status {
  return queryProperty<QDMI_Device_Status>(QDMI_DEVICE_PROPERTY_STATUS);
}
auto Device::getLibraryVersion() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_LIBRARYVERSION);
}
auto Device::getQubitsNum() const -> size_t {
  return queryProperty<size_t>(QDMI_DEVICE_PROPERTY_QUBITSNUM);
}
auto Device::getSites() const -> std::vector<Site> {
  const auto& qdmiSites =
      queryProperty<std::vector<QDMI_Site>>(QDMI_DEVICE_PROPERTY_SITES);
  std::vector<Site> sites;
  sites.reserve(qdmiSites.size());
  std::ranges::transform(
      qdmiSites, std::back_inserter(sites),
      [this](const QDMI_Site& site) -> Site { return {device_, site}; });
  return sites;
}
auto Device::getOperations() const -> std::vector<Operation> {
  const auto& qdmiOperations = queryProperty<std::vector<QDMI_Operation>>(
      QDMI_DEVICE_PROPERTY_OPERATIONS);
  std::vector<Operation> operations;
  operations.reserve(qdmiOperations.size());
  std::ranges::transform(
      qdmiOperations, std::back_inserter(operations),
      [this](const QDMI_Operation& op) -> Operation { return {device_, op}; });
  return operations;
}
auto Device::getCouplingMap() const -> std::vector<std::pair<Site, Site>> {
  const auto& qdmiCouplingMap =
      queryProperty<std::vector<std::pair<QDMI_Site, QDMI_Site>>>(
          QDMI_DEVICE_PROPERTY_COUPLINGMAP);
  std::vector<std::pair<Site, Site>> couplingMap;
  couplingMap.reserve(qdmiCouplingMap.size());
  std::ranges::transform(qdmiCouplingMap, std::back_inserter(couplingMap),
                         [this](const std::pair<QDMI_Site, QDMI_Site>& pair)
                             -> std::pair<Site, Site> {
                           return {Site{device_, pair.first},
                                   Site{device_, pair.second}};
                         });
  return couplingMap;
}
auto Device::getNeedsCalibration() const -> bool {
  return queryProperty<bool>(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION);
}
auto Device::getLengthUnit() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_LENGTHUNIT);
}
auto Device::getLengthScaleFactor() const -> double {
  return queryProperty<double>(QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR);
}
auto Device::getDurationUnit() const -> std::string {
  return queryProperty<std::string>(QDMI_DEVICE_PROPERTY_DURATIONUNIT);
}
auto Device::getDurationScaleFactor() const -> double {
  return queryProperty<double>(QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR);
}
auto Device::getMinAtomDistance() const -> uint64_t {
  return queryProperty<uint64_t>(QDMI_DEVICE_PROPERTY_MINATOMDISTANCE);
}
FoMaC::FoMaC() { qdmi::Driver::get().sessionAlloc(&session_); }
FoMaC::~FoMaC() { qdmi::Driver::get().sessionFree(session_); }
auto FoMaC::queryDevices() -> std::vector<Device> {
  const auto& qdmiDevices = get().queryProperty<std::vector<QDMI_Device>>(
      QDMI_SESSION_PROPERTY_DEVICES);
  std::vector<Device> devices;
  devices.reserve(qdmiDevices.size());
  std::ranges::transform(
      qdmiDevices, std::back_inserter(devices),
      [](const QDMI_Device& dev) -> Device { return Device(dev); });
  return devices;
}
} // namespace fomac
