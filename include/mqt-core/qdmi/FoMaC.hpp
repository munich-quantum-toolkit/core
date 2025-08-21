/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "qdmi/client.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace fomac {
class Site {
  friend class Device;
  /// @brief The associated QDMI_Device object.
  QDMI_Device device_;
  /// @brief The underlying QDMI_Site object.
  QDMI_Site site_;
  /**
   * @brief Constructs a Site object from a QDMI_Site handle.
   * @param site The QDMI_Site handle to wrap.
   */
  explicit Site(QDMI_Device device, QDMI_Site site)
      : device_(device), site_(site) {}

public:
  /// @returns the underlying QDMI_Site object.
  [[nodiscard]] auto getQDMISite() const -> QDMI_Site { return site_; }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator QDMI_Site() const { return site_; }
  /// @see QDMI_SITE_PROPERTY_INDEX
  [[nodiscard]] auto getIndex() const -> size_t;
  /// @see QDMI_SITE_PROPERTY_T1
  [[nodiscard]] auto getT1() const -> uint64_t;
  /// @see QDMI_SITE_PROPERTY_T2
  [[nodiscard]] auto getT2() const -> uint64_t;
  /// @see QDMI_SITE_PROPERTY_NAME
  [[nodiscard]] auto getName() const -> std::string;
  /// @see QDMI_SITE_PROPERTY_XCOORDINATE
  [[nodiscard]] auto getXCoordinate() const -> int64_t;
  /// @see QDMI_SITE_PROPERTY_YCOORDINATE
  [[nodiscard]] auto getYCoordinate() const -> int64_t;
  /// @see QDMI_SITE_PROPERTY_ZCOORDINATE
  [[nodiscard]] auto getZCoordinate() const -> int64_t;
  /// @see QDMI_SITE_PROPERTY_ISZONE
  [[nodiscard]] auto isZone() const -> bool;
  /// @see QDMI_SITE_PROPERTY_XEXTENT
  [[nodiscard]] auto getXExtent() const -> uint64_t;
  /// @see QDMI_SITE_PROPERTY_YEXTENT
  [[nodiscard]] auto getYExtent() const -> uint64_t;
  /// @see QDMI_SITE_PROPERTY_ZEXTENT
  [[nodiscard]] auto getZExtent() const -> uint64_t;
  /// @see QDMI_SITE_PROPERTY_MODULEINDEX
  [[nodiscard]] auto getModuleIndex() const -> size_t;
  /// @see QDMI_SITE_PROPERTY_SUBMODULEINDEX
  [[nodiscard]] auto getSubmoduleIndex() const -> size_t;
};
class Operation {
  friend class Device;
  /// @brief The associated QDMI_Device object.
  QDMI_Device device_;
  /// @brief The underlying QDMI_Operation object.
  QDMI_Operation operation_;
  /**
   * @brief Constructs an Operation object from a QDMI_Operation handle.
   * @param operation The QDMI_Operation handle to wrap.
   */
  explicit Operation(QDMI_Operation operation) : operation_(operation) {}

public:
  /// @returns the underlying QDMI_Operation object.
  [[nodiscard]] auto getQDMIOperation() const -> QDMI_Operation {
    return operation_;
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator QDMI_Operation() const { return operation_; }
  /// @see QDMI_OPERATION_PROPERTY_NAME
  [[nodiscard]] auto getName() -> std::string;
  /// @see QDMI_OPERATION_PROPERTY_QUBITSNUM
  [[nodiscard]] auto getQubitsNum() -> size_t;
  /// @see QDMI_OPERATION_PROPERTY_PARAMETERSNUM
  [[nodiscard]] auto getParametersNum() -> size_t;
  /// @see QDMI_OPERATION_PROPERTY_DURATION
  [[nodiscard]] auto getDuration() -> uint64_t;
  /// @see QDMI_OPERATION_PROPERTY_FIDELITY
  [[nodiscard]] auto getFidelity() -> double;
  /// @see QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS
  [[nodiscard]] auto getInteractionRadius() -> uint64_t;
  /// @see QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS
  [[nodiscard]] auto getBlockingRadius() -> uint64_t;
  /// @see QDMI_OPERATION_PROPERTY_IDLINGFIDELITY
  [[nodiscard]] auto getIdlingFidelity() -> double;
  /// @see QDMI_OPERATION_PROPERTY_ISZONED
  [[nodiscard]] auto isZoned() -> bool;
  /// @see QDMI_OPERATION_PROPERTY_SITES
  [[nodiscard]] auto getSites() -> std::vector<Site>;
  /// @see QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED
  [[nodiscard]] auto getMeanShuttlingSpeed() -> uint64_t;
};
class Device {
  /// @brief The underlying QDMI_Device object.
  QDMI_Device device_;
  /**
   * @brief Constructs a Device object from a QDMI_Device handle.
   * @param device The QDMI_Device handle to wrap.
   */
  explicit Device(QDMI_Device device) : device_(device) {}

public:
  /// @returns the underlying QDMI_Device object.
  [[nodiscard]] auto getQDMIDevice() const -> QDMI_Device { return device_; }
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator QDMI_Device() const { return device_; }
  /// @see QDMI_DEVICE_PROPERTY_NAME
  [[nodiscard]] auto getName() -> std::string;
  /// @see QDMI_DEVICE_PROPERTY_VERSION
  [[nodiscard]] auto getVersion() -> std::string;
  /// @see QDMI_DEVICE_PROPERTY_STATUS
  [[nodiscard]] auto getStatus() -> QDMI_Device_Status;
  /// @see QDMI_DEVICE_PROPERTY_LIBRARYVERSION
  [[nodiscard]] auto getLibraryVersion() -> std::string;
  /// @see QDMI_DEVICE_PROPERTY_QUBITSNUM
  [[nodiscard]] auto getQubitsNum() -> size_t;
  /// @see QDMI_DEVICE_PROPERTY_SITES
  [[nodiscard]] auto getSites() -> std::vector<Site>;
  /// @see QDMI_DEVICE_PROPERTY_OPERATIONS
  [[nodiscard]] auto getOperations() -> std::vector<Operation>;
  /// @see QDMI_DEVICE_PROPERTY_COUPLINGMAP
  [[nodiscard]] auto getCouplingMap() -> std::vector<std::pair<Site, Site>>;
  /// @see QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION
  [[nodiscard]] auto getNeedsCalibration() -> bool;
  /// @see QDMI_DEVICE_PROPERTY_LENGTHUNIT
  [[nodiscard]] auto getLengthUnit() -> std::string;
  /// @see QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR
  [[nodiscard]] auto getLengthScaleFactor() -> double;
  /// @see QDMI_DEVICE_PROPERTY_DURATIONUNIT
  [[nodiscard]] auto getDurationUnit() -> std::string;
  /// @see QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR
  [[nodiscard]] auto getDurationScaleFactor() -> double;
  /// @see QDMI_DEVICE_PROPERTY_MINATOMDISTANCE
  [[nodiscard]] auto getMinAtomDistance() -> uint64_t;
};
} // namespace fomac
