/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/Target.h"
#include "qdmi/Client.hpp"
#include "qdmi/common/Common.hpp"
#include "qdmi/driver/Driver.hpp"

#include "support/Diagnostics.hpp"
#include "support/TestSupport.hpp"

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "qdmi/client.h"
#include "qdmi/device.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using mlir::CompilerTarget;

static qdmi::Device openScDevice(const char* filename) {
  const auto config = nlohmann::json{
      {
          "device-config",
          {{"file", std::string{MQT_CORE_MLIR_SC_CONFIG_DIR} + "/" + filename}},
      },
  };
  return ::mqt::test::value(
      qdmi::builtin_driver::openDevice("mqt.sc.default", config.dump()));
}

[[nodiscard]] static const CompilerTarget::OperationCapability&
findOperation(const CompilerTarget& target, const llvm::StringRef name) {
  const auto* const found =
      llvm::find_if(target.operations(),
                    [&](const auto& op) { return op.canonicalName() == name; });
  assert(found != target.operations().end() && "Target operation not found");
  return *found;
}

TEST(CompilerQDMIAdapterTest, SnapshotsIQMCalibrationAndLifetime) {
  const auto target = ::mqt::test::value([] {
    const auto device =
        ::mqt::test::value(qdmi::Session::openDevice("mqt.sc.iqm.garnet"));
    return mlir::compilerTargetFromDevice(device);
  }());

  ASSERT_TRUE(target.name());
  EXPECT_EQ(*target.name(), "IQM Garnet");
  EXPECT_EQ(target.numSites(), 20);
  EXPECT_EQ(target.connectivityKind(),
            CompilerTarget::Connectivity::Kind::Explicit);
  EXPECT_EQ(target.couplings().size(), 30);

  ASSERT_TRUE(target.durationUnit());
  EXPECT_EQ(target.durationUnit()->unit(), "us");
  EXPECT_DOUBLE_EQ(target.durationUnit()->scaleFactor(), 0.001);

  ASSERT_EQ(target.sites().size(), 20);
  ASSERT_TRUE(target.sites().front().name());
  EXPECT_EQ(*target.sites().front().name(), "QB1");
  EXPECT_EQ(target.sites().front().t1(), 26626);
  EXPECT_EQ(target.sites().front().t2(), 8376);

  ASSERT_EQ(target.operations().size(), 3);
  const auto& r = findOperation(target, "r");
  const auto& cz = findOperation(target, "cz");
  const auto& measure = findOperation(target, "measure");
  EXPECT_EQ(r.siteTuples().size(), 20);
  EXPECT_EQ(cz.siteTuples().size(), 30);
  EXPECT_EQ(measure.siteTuples().size(), 20);
  for (const auto& operation : target.operations()) {
    EXPECT_FALSE(operation.duration());
    for (const auto& tuple : operation.siteTuples()) {
      EXPECT_FALSE(tuple.duration());
      EXPECT_TRUE(tuple.fidelity());
    }
  }

  EXPECT_EQ(target.supportsOperation("r", 1, 2), true);
  EXPECT_EQ(target.supportsOperation("cz", 2, 0), true);
  EXPECT_EQ(target.supportsOperation("measure", 1, 0), true);
  EXPECT_EQ(target.supportsOperation("rx", 1, 1), false);
  ASSERT_TRUE(target.synthesisBasis());
  EXPECT_EQ(target.synthesisBasis()->singleQubit,
            CompilerTarget::SingleQubitBasis::R);
  EXPECT_EQ(target.synthesisBasis()->entangler, CompilerTarget::GateKind::CZ);
}

TEST(CompilerQDMIAdapterTest, QueriesNamesAndSiteIndicesOncePerSnapshot) {
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.sc.default"));
  auto* library = &const_cast<qdmi::DeviceLibrary&>(
      static_cast<QDMI_Device>(device)->getLibrary());
  static thread_local decltype(QDMI_device_session_query_site_property)*
      querySite = nullptr;
  static thread_local decltype(QDMI_device_session_query_operation_property)*
      queryOperation = nullptr;
  static thread_local size_t indexQueries = 0;
  static thread_local size_t nameQueries = 0;
  querySite = library->device_session_query_site_property;
  queryOperation = library->device_session_query_operation_property;
  const auto restore = llvm::make_scope_exit([&] {
    library->device_session_query_site_property = querySite;
    library->device_session_query_operation_property = queryOperation;
  });
  library->device_session_query_site_property =
      [](QDMI_Device_Session session, QDMI_Site site,
         QDMI_Site_Property property, size_t size, void* value,
         size_t* sizeRet) {
        indexQueries += property == QDMI_SITE_PROPERTY_INDEX;
        return querySite(session, site, property, size, value, sizeRet);
      };
  library->device_session_query_operation_property =
      [](QDMI_Device_Session session, QDMI_Operation operation, size_t numSites,
         const QDMI_Site* sites, size_t numParams, const double* params,
         QDMI_Operation_Property property, size_t size, void* value,
         size_t* sizeRet) {
        nameQueries += property == QDMI_OPERATION_PROPERTY_NAME;
        return queryOperation(session, operation, numSites, sites, numParams,
                              params, property, size, value, sizeRet);
      };
  for (int snapshot = 0; snapshot < 2; ++snapshot) {
    indexQueries = 0;
    nameQueries = 0;
    const auto target =
        ::mqt::test::value(mlir::compilerTargetFromDevice(device));
    /// Bound provider calls independently of coupling and operation counts.
    EXPECT_EQ(indexQueries, target.numSites());
    EXPECT_EQ(nameQueries, 2 * target.operations().size());
    EXPECT_EQ(target.numSites(), 100);
    EXPECT_EQ(target.operations().size(), 3);
  }
}

TEST(CompilerQDMIAdapterTest, ReturnsProviderQueryFailuresAsErrors) {
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.sc.default"));
  auto* library = &const_cast<qdmi::DeviceLibrary&>(
      static_cast<QDMI_Device>(device)->getLibrary());
  static thread_local decltype(QDMI_device_session_query_device_property)*
      queryDevice = nullptr;
  static thread_local decltype(QDMI_device_session_query_site_property)*
      querySite = nullptr;
  static thread_local decltype(QDMI_device_session_query_operation_property)*
      queryOperation = nullptr;
  static thread_local std::optional<QDMI_Device_Property> failingDeviceProperty;
  static thread_local std::optional<QDMI_Site_Property> failingSiteProperty;
  static thread_local std::optional<QDMI_Operation_Property>
      failingOperationProperty;
  static thread_local bool failSiteCalibration = false;
  static thread_local int queryFailure = QDMI_ERROR_PERMISSIONDENIED;
  queryFailure = QDMI_ERROR_PERMISSIONDENIED;
  queryDevice = library->device_session_query_device_property;
  querySite = library->device_session_query_site_property;
  queryOperation = library->device_session_query_operation_property;
  const auto restore = llvm::make_scope_exit([&] {
    library->device_session_query_device_property = queryDevice;
    library->device_session_query_site_property = querySite;
    library->device_session_query_operation_property = queryOperation;
  });
  library->device_session_query_device_property =
      [](QDMI_Device_Session session, QDMI_Device_Property property,
         size_t size, void* value, size_t* sizeRet) {
        return property == failingDeviceProperty
                   ? queryFailure
                   : queryDevice(session, property, size, value, sizeRet);
      };
  library->device_session_query_site_property =
      [](QDMI_Device_Session session, QDMI_Site site,
         QDMI_Site_Property property, size_t size, void* value,
         size_t* sizeRet) {
        return property == failingSiteProperty
                   ? queryFailure
                   : querySite(session, site, property, size, value, sizeRet);
      };
  library->device_session_query_operation_property =
      [](QDMI_Device_Session session, QDMI_Operation operation, size_t numSites,
         const QDMI_Site* sites, size_t numParams, const double* params,
         QDMI_Operation_Property property, size_t size, void* value,
         size_t* sizeRet) {
        return property == failingOperationProperty &&
                       (!failSiteCalibration || numSites != 0)
                   ? queryFailure
                   : queryOperation(session, operation, numSites, sites,
                                    numParams, params, property, size, value,
                                    sizeRet);
      };

  const auto expectFailure = [&](auto property) {
    SCOPED_TRACE(qdmi::toString(property));
    const auto error = ::mqt::test::diagnostic(
        [&] { return mlir::compilerTargetFromDevice(device); });
    ASSERT_TRUE(error);
    EXPECT_EQ(error->status, QDMI_ERROR_PERMISSIONDENIED);
    EXPECT_NE(error->message.find(qdmi::toString(property)), std::string::npos);
  };
  for (const auto property : {
           QDMI_DEVICE_PROPERTY_NAME,
           QDMI_DEVICE_PROPERTY_CUSTOM1,
           QDMI_DEVICE_PROPERTY_SITES,
           QDMI_DEVICE_PROPERTY_QUBITSNUM,
           QDMI_DEVICE_PROPERTY_COUPLINGMAP,
           QDMI_DEVICE_PROPERTY_OPERATIONS,
           QDMI_DEVICE_PROPERTY_DURATIONUNIT,
           QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR,
       }) {
    failingDeviceProperty = property;
    expectFailure(property);
  }
  failingDeviceProperty = QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS;
  ::mqt::test::DiagnosticCapture environmentDiagnostics;
  auto environment = mlir::targetEnvironmentFromDevice(device);
  ASSERT_FALSE(succeeded(environment));
  EXPECT_NE(environmentDiagnostics.error->message.find(
                qdmi::toString(QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS)),
            std::string::npos);
  failingDeviceProperty.reset();
  for (const auto property : {
           QDMI_SITE_PROPERTY_ISZONE,
           QDMI_SITE_PROPERTY_INDEX,
           QDMI_SITE_PROPERTY_NAME,
           QDMI_SITE_PROPERTY_T1,
           QDMI_SITE_PROPERTY_T2,
       }) {
    failingSiteProperty = property;
    expectFailure(property);
  }
  failingSiteProperty.reset();
  for (const auto property : {
           QDMI_OPERATION_PROPERTY_ISZONED,
           QDMI_OPERATION_PROPERTY_QUBITSNUM,
           QDMI_OPERATION_PROPERTY_NAME,
           QDMI_OPERATION_PROPERTY_CUSTOM1,
           QDMI_OPERATION_PROPERTY_SITES,
           QDMI_OPERATION_PROPERTY_DURATION,
           QDMI_OPERATION_PROPERTY_FIDELITY,
           QDMI_OPERATION_PROPERTY_PARAMETERSNUM,
       }) {
    failingOperationProperty = property;
    expectFailure(property);
  }
  failSiteCalibration = true;
  for (const auto property :
       {QDMI_OPERATION_PROPERTY_DURATION, QDMI_OPERATION_PROPERTY_FIDELITY}) {
    failingOperationProperty = property;
    expectFailure(property);
  }
  failingOperationProperty.reset();
  failSiteCalibration = false;
  queryFailure = QDMI_ERROR_NOTSUPPORTED;
  failingDeviceProperty = QDMI_DEVICE_PROPERTY_COUPLINGMAP;
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return mlir::compilerTargetFromDevice(device); }),
            ::mqt::ErrorCategory::InvalidArgument);
  failingDeviceProperty = QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR;
  const auto unscaled =
      ::mqt::test::value(mlir::compilerTargetFromDevice(device));
  ASSERT_TRUE(unscaled.durationUnit());
  EXPECT_DOUBLE_EQ(unscaled.durationUnit()->scaleFactor(), 1.);
  failingDeviceProperty = QDMI_DEVICE_PROPERTY_DURATIONUNIT;
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return mlir::compilerTargetFromDevice(device); }),
            ::mqt::ErrorCategory::InvalidArgument);
  failingDeviceProperty.reset();
  failingSiteProperty = QDMI_SITE_PROPERTY_NAME;
  const auto unnamed =
      ::mqt::test::value(mlir::compilerTargetFromDevice(device));
  EXPECT_FALSE(unnamed.sites().front().name());
  failingSiteProperty.reset();
  failingOperationProperty = QDMI_OPERATION_PROPERTY_SITES;
  EXPECT_EQ(::mqt::test::errorKind(
                [&] { return mlir::compilerTargetFromDevice(device); }),
            ::mqt::ErrorCategory::InvalidArgument);
  failingOperationProperty = QDMI_OPERATION_PROPERTY_DURATION;
  const auto untimed =
      ::mqt::test::value(mlir::compilerTargetFromDevice(device));
  for (const auto& operation : untimed.operations()) {
    EXPECT_FALSE(operation.duration());
    for (const auto& tuple : operation.siteTuples()) {
      EXPECT_FALSE(tuple.duration());
    }
  }
  failingOperationProperty.reset();
  EXPECT_EQ(
      ::mqt::test::value(mlir::compilerTargetFromDevice(device)).numSites(),
      100);
}

TEST(CompilerQDMIAdapterTest, RejectsMalformedProviderMetadataAndRecovers) {
  enum class InvalidMetadata : uint8_t {
    None,
    SiteIndex,
    SiteName,
    SiteT1,
    ZoneSite,
    QubitCount,
    DurationScale,
    ZonedOperation,
    OperationName,
    OperationArity,
    NondivisibleSites,
    RepeatedPairQubit,
    RepeatedTupleQubit,
    DuplicateSites,
    OperationFidelity,
    SiteFidelity,
  };
  static thread_local InvalidMetadata invalid = InvalidMetadata::None;
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.sc.default"));
  auto* library = &const_cast<qdmi::DeviceLibrary&>(
      static_cast<QDMI_Device>(device)->getLibrary());
  static thread_local auto queryDevice =
      library->device_session_query_device_property;
  static thread_local auto querySite =
      library->device_session_query_site_property;
  static thread_local auto queryOperation =
      library->device_session_query_operation_property;
  library->device_session_query_device_property =
      [](QDMI_Device_Session session, QDMI_Device_Property property,
         size_t size, void* value, size_t* sizeRet) -> int {
    const auto status = queryDevice(session, property, size, value, sizeRet);
    if (status == QDMI_SUCCESS && value != nullptr) {
      if (invalid == InvalidMetadata::QubitCount &&
          property == QDMI_DEVICE_PROPERTY_QUBITSNUM) {
        ++*static_cast<size_t*>(value);
      } else if (invalid == InvalidMetadata::DurationScale &&
                 property == QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR) {
        *static_cast<double*>(value) = -1.;
      }
    }
    return status;
  };
  library->device_session_query_site_property =
      [](QDMI_Device_Session session, QDMI_Site site,
         QDMI_Site_Property property, size_t size, void* value,
         size_t* sizeRet) -> int {
    if (invalid == InvalidMetadata::SiteName) {
      ADD_STRING_PROPERTY(QDMI_SITE_PROPERTY_NAME, "", property, size, value,
                          sizeRet)
    }
    if (invalid == InvalidMetadata::ZoneSite) {
      ADD_SINGLE_VALUE_PROPERTY(QDMI_SITE_PROPERTY_ISZONE, bool, true, property,
                                size, value, sizeRet)
    }
    const auto status =
        querySite(session, site, property, size, value, sizeRet);
    if (status == QDMI_SUCCESS && value != nullptr) {
      if (invalid == InvalidMetadata::SiteIndex &&
          property == QDMI_SITE_PROPERTY_INDEX) {
        *static_cast<size_t*>(value) = std::numeric_limits<size_t>::max();
      } else if (invalid == InvalidMetadata::SiteT1 &&
                 property == QDMI_SITE_PROPERTY_T1) {
        *static_cast<uint64_t*>(value) = 0;
      }
    }
    return status;
  };
  library->device_session_query_operation_property =
      [](QDMI_Device_Session session, QDMI_Operation operation, size_t numSites,
         const QDMI_Site* sites, size_t numParams, const double* params,
         QDMI_Operation_Property property, size_t size, void* value,
         size_t* sizeRet) -> int {
    if (invalid == InvalidMetadata::OperationName) {
      ADD_STRING_PROPERTY(QDMI_OPERATION_PROPERTY_NAME, "", property, size,
                          value, sizeRet)
    }
    if (invalid == InvalidMetadata::ZonedOperation) {
      ADD_SINGLE_VALUE_PROPERTY(QDMI_OPERATION_PROPERTY_ISZONED, bool, true,
                                property, size, value, sizeRet)
    }
    const auto status =
        queryOperation(session, operation, numSites, sites, numParams, params,
                       property, size, value, sizeRet);
    if (status == QDMI_SUCCESS && value != nullptr) {
      if (invalid == InvalidMetadata::OperationArity &&
          property == QDMI_OPERATION_PROPERTY_QUBITSNUM) {
        *static_cast<size_t*>(value) = 0;
      } else if (property == QDMI_OPERATION_PROPERTY_QUBITSNUM &&
                 (invalid == InvalidMetadata::NondivisibleSites ||
                  invalid == InvalidMetadata::RepeatedPairQubit ||
                  invalid == InvalidMetadata::RepeatedTupleQubit)) {
        *static_cast<size_t*>(value) =
            invalid == InvalidMetadata::RepeatedPairQubit ? 2 : 4;
        if (invalid == InvalidMetadata::NondivisibleSites) {
          *static_cast<size_t*>(value) = 3;
        }
      } else if ((invalid == InvalidMetadata::DuplicateSites ||
                  invalid == InvalidMetadata::RepeatedPairQubit ||
                  invalid == InvalidMetadata::RepeatedTupleQubit) &&
                 property == QDMI_OPERATION_PROPERTY_SITES &&
                 size >= 2 * sizeof(QDMI_Site)) {
        auto* output = static_cast<QDMI_Site*>(value);
        output[1] = output[0];
      } else if (property == QDMI_OPERATION_PROPERTY_FIDELITY &&
                 (invalid == InvalidMetadata::OperationFidelity ||
                  (invalid == InvalidMetadata::SiteFidelity &&
                   numSites != 0))) {
        *static_cast<double*>(value) = -1.;
      }
    }
    return status;
  };

  for (const auto scenario : {
           InvalidMetadata::SiteIndex,
           InvalidMetadata::SiteName,
           InvalidMetadata::SiteT1,
           InvalidMetadata::ZoneSite,
           InvalidMetadata::QubitCount,
           InvalidMetadata::DurationScale,
           InvalidMetadata::ZonedOperation,
           InvalidMetadata::OperationName,
           InvalidMetadata::OperationArity,
           InvalidMetadata::NondivisibleSites,
           InvalidMetadata::RepeatedPairQubit,
           InvalidMetadata::RepeatedTupleQubit,
           InvalidMetadata::DuplicateSites,
           InvalidMetadata::OperationFidelity,
           InvalidMetadata::SiteFidelity,
       }) {
    SCOPED_TRACE(static_cast<int>(scenario));
    invalid = scenario;
    EXPECT_EQ(::mqt::test::errorKind(
                  [&] { return mlir::compilerTargetFromDevice(device); }),
              ::mqt::ErrorCategory::InvalidArgument);
  }
  invalid = InvalidMetadata::None;
  EXPECT_EQ(
      ::mqt::test::value(mlir::compilerTargetFromDevice(device)).numSites(),
      100);
}

TEST(CompilerQDMIAdapterTest, InfersDDSIMTargetFacts) {
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.ddsim.default"));
  const auto target =
      ::mqt::test::value(mlir::compilerTargetFromDevice(device));

  EXPECT_EQ(target.numSites(), 65535);
  EXPECT_EQ(target.connectivityKind(),
            CompilerTarget::Connectivity::Kind::AllToAll);
  EXPECT_EQ(target.nativeOperationsKind(),
            CompilerTarget::NativeOperations::Kind::Explicit);
  const auto& gphase = findOperation(target, "gphase");
  EXPECT_EQ(gphase.arity().kind(),
            CompilerTarget::OperationCapability::Arity::Kind::Fixed);
  EXPECT_EQ(gphase.arity().value(), 0);
  for (const auto [name, minimum] :
       std::initializer_list<std::pair<llvm::StringRef, size_t>>{
           {"id", 1},
           {"h", 1},
           {"rx", 1},
           {"swap", 2},
           {"rxx", 2},
           {"rccx", 3},
       }) {
    const auto& operation = findOperation(target, name);
    EXPECT_EQ(operation.arity().kind(),
              CompilerTarget::OperationCapability::Arity::Kind::Variadic)
        << name.str();
    EXPECT_EQ(operation.arity().value(), minimum) << name.str();
    EXPECT_TRUE(operation.siteTuples().empty()) << name.str();
    EXPECT_TRUE(
        target.supportsOperation(name, minimum, operation.numParameters()))
        << name.str();
    EXPECT_TRUE(
        target.supportsOperation(name, minimum + 4, operation.numParameters()))
        << name.str();
    std::vector<CompilerTarget::SiteId> sites(minimum + 4);
    std::iota(sites.begin(), sites.end(), 0);
    EXPECT_TRUE(target.supportsOperation(name, minimum + 4,
                                         operation.numParameters(), sites))
        << name.str();
  }
  EXPECT_TRUE(target.supportsOperation("gphase", 0, 1));
  EXPECT_EQ(target.supportsOperation("h", 1, 0), true);
  EXPECT_EQ(target.supportsOperation("cx", 2, 0), true);
  EXPECT_EQ(target.supportsOperation("cswap", 3, 0), true);
  EXPECT_EQ(target.supportsOperation("measure", 1, 0), true);
  EXPECT_EQ(target.supportsOperation("reset", 1, 0), true);
  EXPECT_EQ(target.supportsOperation("barrier", 0, 0), false);
}

TEST(CompilerQDMIAdapterTest, ListsRegisteredDeviceIds) {
  const auto deviceIds = ::mqt::test::value(mlir::registeredQDMIDeviceIds());
  EXPECT_TRUE(llvm::is_contained(deviceIds, "mqt.ddsim.default"));
}

TEST(CompilerQDMIAdapterTest, ConvertsUnknownDeviceFailureToError) {
  ::mqt::test::DiagnosticCapture targetDiagnostics;
  auto target = mlir::compilerTargetFromDeviceId("mqt.unknown.device");
  ASSERT_FALSE(succeeded(target));
  const auto message = targetDiagnostics.error->message;
  EXPECT_NE(message.find("mqt.unknown.device"), std::string::npos);
}

TEST(CompilerQDMIAdapterTest, RejectsNonhomogeneousOperationSupport) {
  const auto device = openScDevice("heterogeneous-sc.json");
  ::mqt::test::DiagnosticCapture targetDiagnostics;
  auto target = mlir::compilerTargetFromDevice(device);
  ASSERT_FALSE(succeeded(target));
  const auto message = targetDiagnostics.error->message;
  EXPECT_NE(message.find("homogeneous"), std::string::npos);
  EXPECT_NE(message.find("all topology edges"), std::string::npos);
}

TEST(CompilerQDMIAdapterTest, SnapshotsHomogeneousHigherArityOperation) {
  const auto device = openScDevice("higher-arity-sc.json");
  const auto target =
      ::mqt::test::value(mlir::compilerTargetFromDevice(device));

  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0));
  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0, {0, 1, 2}));
  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0, {2, 1, 0}));
  EXPECT_FALSE(target.supportsOperation("ccnot", 3, 0, {0, 1, 3}));
}

TEST(CompilerQDMIAdapterTest, PreservesOneWayDirectionalOperationSupport) {
  const auto device = openScDevice("directional-one-way-sc.json");
  const auto target =
      ::mqt::test::value(mlir::compilerTargetFromDevice(device));

  ASSERT_EQ(target.couplings().size(), 1U);
  const auto& cx = findOperation(target, "cx");
  ASSERT_EQ(cx.siteTuples().size(), 1U);
  EXPECT_EQ(cx.siteTuples()[0].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{0, 1}));
  EXPECT_FALSE(cx.siteTuples()[0].duration());
  EXPECT_FALSE(cx.siteTuples()[0].fidelity());
  EXPECT_TRUE(target.supportsOperation("cx", 2, 0, {0, 1}));
  EXPECT_FALSE(target.supportsOperation("cx", 2, 0, {1, 0}));
  ASSERT_TRUE(target.synthesisBasis());
  EXPECT_EQ(target.synthesisBasis()->entangler, CompilerTarget::GateKind::CX);
}

TEST(CompilerQDMIAdapterTest, OmitsOperationsWithNoSupportedPlacements) {
  const auto device = ::mqt::test::value(qdmi::builtin_driver::openDevice(
      "mqt.sc.default", R"({"device-config":{"inline":{
    "schema-version": 1,
    "name": "Unavailable operation",
    "numQubits": 1,
    "durationUnit": {"unit": "ns", "scaleFactor": 1},
    "qubitProperties": {"defaults": {}, "overrides": []},
    "couplings": [],
    "operations": [
      {"name": "x", "numQubits": 1, "numParameters": 0, "sites": []}
    ]
  }}})"));
  const auto target =
      ::mqt::test::value(mlir::compilerTargetFromDevice(device));
  EXPECT_EQ(target.nativeOperationsKind(),
            CompilerTarget::NativeOperations::Kind::Explicit);
  EXPECT_TRUE(target.operations().empty());
  EXPECT_FALSE(target.supportsOperation("x", 1, 0, {0}));
}

TEST(CompilerQDMIAdapterTest,
     PreservesDirectionalCalibrationWhenBothOrientationsExist) {
  const auto device = openScDevice("directional-two-way-sc.json");
  const auto target =
      ::mqt::test::value(mlir::compilerTargetFromDevice(device));

  ASSERT_EQ(target.couplings().size(), 1);
  const auto& cx = findOperation(target, "cx");
  EXPECT_TRUE(target.supportsOperation("cx", 2, 0, {0, 1}));
  EXPECT_TRUE(target.supportsOperation("cx", 2, 0, {1, 0}));
  ASSERT_EQ(cx.siteTuples().size(), 2);
  EXPECT_EQ(cx.siteTuples()[0].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{0, 1}));
  EXPECT_DOUBLE_EQ(*cx.siteTuples()[0].fidelity(), 0.91);
  EXPECT_EQ(cx.siteTuples()[1].sites(),
            (llvm::ArrayRef<CompilerTarget::SiteId>{1, 0}));
  EXPECT_DOUBLE_EQ(*cx.siteTuples()[1].fidelity(), 0.92);
}

TEST(CompilerQDMIAdapterTest,
     SelectsPayloadByPreferenceAndIncludesMaximalCapabilities) {
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.ddsim.default"));
  auto* library = &const_cast<qdmi::DeviceLibrary&>(
      static_cast<QDMI_Device>(device)->getLibrary());
  static thread_local decltype(QDMI_device_session_query_device_property)*
      query = nullptr;
  static thread_local std::vector<QDMI_Program_Format> formats;
  query = library->device_session_query_device_property;
  const auto restore = llvm::make_scope_exit(
      [&] { library->device_session_query_device_property = query; });
  library->device_session_query_device_property =
      [](QDMI_Device_Session session, QDMI_Device_Property property,
         size_t size, void* value, size_t* sizeRet) -> int {
    if (property != QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS) {
      return query(session, property, size, value, sizeRet);
    }
    const size_t required = formats.size() * sizeof(QDMI_Program_Format);
    if (sizeRet != nullptr) {
      *sizeRet = required;
    }
    if (value != nullptr) {
      if (size < required) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      std::memcpy(value, formats.data(), required);
    }
    return QDMI_SUCCESS;
  };
  formats = {
      QDMI_PROGRAM_FORMAT_QIRBASESTRING,
      QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
      QDMI_PROGRAM_FORMAT_QASM3,
      QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
      QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
  };
  while (!formats.empty()) {
    const auto environment =
        ::mqt::test::value(mlir::targetEnvironmentFromDevice(device));
    const auto baseline = ::mqt::test::value(
        mlir::payloadSpecificationForProgramFormat(formats.back()));
    EXPECT_EQ(environment.payloadSpecification().format(), baseline.format());
    EXPECT_EQ(environment.payloadSpecification().capabilities(),
              baseline.capabilities());
    auto compiled = ::mqt::test::value(mlir::compileProgram(
        mlir::OpenQASMProgram(
            "OPENQASM 3.0; include \"stdgates.inc\"; qubit q; "
            "bit c; x q; c = measure q;"),
        device, formats.back()));
    EXPECT_EQ(compiled.programFormat(), formats.back());
    auto job = ::mqt::test::value(mlir::submitProgram(device, compiled, 2));
    ASSERT_TRUE(::mqt::test::value(job.wait()));
    EXPECT_EQ(::mqt::test::value(job.getShots()),
              (std::vector<std::string>{"1", "1"}));
    formats.pop_back();
  }
  EXPECT_TRUE(failed(mlir::targetEnvironmentFromDevice(device)));
  EXPECT_TRUE(failed(mlir::compileProgram(
      mlir::OpenQASMProgram("OPENQASM 3.0; qubit q;"), device)));
  formats = {QDMI_PROGRAM_FORMAT_QASM2};
  EXPECT_TRUE(failed(
      mlir::targetEnvironmentFromDevice(device, QDMI_PROGRAM_FORMAT_QASM2)));
  formats = {QDMI_PROGRAM_FORMAT_QASM3};
  EXPECT_TRUE(failed(mlir::targetEnvironmentFromDevice(
      device, QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE)));
  EXPECT_TRUE(failed(
      mlir::payloadSpecificationForProgramFormat(QDMI_PROGRAM_FORMAT_QASM2)));

  const auto adaptive =
      ::mqt::test::value(mlir::payloadSpecificationForProgramFormat(
          QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE));
  ASSERT_EQ(adaptive.capabilities().size(), 11);
  EXPECT_EQ(adaptive.capabilities().front().id, "forward-branching");
  const auto qasm = ::mqt::test::value(
      mlir::payloadSpecificationForProgramFormat(QDMI_PROGRAM_FORMAT_QASM3));
  EXPECT_EQ(qasm.format().version, "3.1.0");
  ASSERT_EQ(qasm.capabilities().size(), 4);
  EXPECT_EQ(qasm.capabilities()[1].id, "counted-iteration");
  EXPECT_EQ(qasm.capabilities()[2].id, "conditional-loop");
  EXPECT_EQ(qasm.capabilities()[3].id, "multiway-branching");
}

TEST(CompilerQDMIAdapterTest,
     CompatibilityIgnoresCalibrationButPreservesLegality) {
  const auto makeTarget = [](bool calibrated,
                             const std::vector<CompilerTarget::SiteId>& ids,
                             std::vector<CompilerTarget::SiteId> operands,
                             double timeScale = 1.) {
    std::vector<CompilerTarget::Site> sites;
    sites.reserve(ids.size());
    for (const auto id : ids) {
      sites.push_back(::mqt::test::value(CompilerTarget::Site::create(
          id, calibrated ? "new name" : "old name", calibrated ? 200 : 100)));
    }
    const auto tuple = ::mqt::test::value(CompilerTarget::SiteTuple::create(
        std::move(operands), calibrated ? 50 : 10, calibrated ? 0.99 : 0.9));
    const auto operation =
        ::mqt::test::value(CompilerTarget::OperationCapability::create(
            "cx", 2, 0, {tuple}, calibrated ? 50 : 10,
            calibrated ? 0.99 : 0.9));
    return ::mqt::test::value(CompilerTarget::create(
        calibrated ? "updated device" : "original device", std::move(sites),
        CompilerTarget::Connectivity::allToAll(),
        CompilerTarget::NativeOperations::fromOperations({operation}),
        ::mqt::test::value(
            CompilerTarget::DurationUnit::create("ns", timeScale))));
  };
  const auto payload = ::mqt::test::value(
      mlir::payloadSpecificationForProgramFormat(QDMI_PROGRAM_FORMAT_QASM3));
  const mlir::TargetEnvironment original(makeTarget(false, {2, 5}, {2, 5}),
                                         payload);
  const mlir::TargetEnvironment calibrated(makeTarget(true, {2, 5}, {2, 5}),
                                           payload);
  EXPECT_FALSE(failed(mlir::validateTargetCompatibility(original, calibrated)));
  for (const auto& target : {
           makeTarget(false, {5, 2}, {2, 5}),
           makeTarget(false, {2, 5}, {5, 2}),
           makeTarget(false, {2, 5}, {2, 5}, 2.),
       }) {
    const mlir::TargetEnvironment changed(target, payload);
    const auto error = ::mqt::test::errorMessage(
        [&] { return mlir::validateTargetCompatibility(original, changed); });
    EXPECT_NE(error.find("recompile for this device"), std::string::npos);
  }
  const auto constrained =
      ::mqt::test::value(mlir::PayloadSpecification::create(
          payload.format(),
          {
              {
                  .id = "forward-branching",
                  .value = 0,
                  .constraints =
                      {
                          {.id = "max-control-flow-nesting-depth", .value = 1},
                      },
              },
          }));
  EXPECT_TRUE(failed(mlir::validateTargetCompatibility(
      original, mlir::TargetEnvironment(original.target(), constrained))));
}

TEST(CompilerQDMIAdapterTest,
     CompilationCreatesNoJobAndSubmissionChecksContract) {
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.ddsim.default"));
  auto* library = &const_cast<qdmi::DeviceLibrary&>(
      static_cast<QDMI_Device>(device)->getLibrary());
  static thread_local decltype(QDMI_device_session_create_device_job)*
      createJob = nullptr;
  static thread_local size_t creations = 0;
  static thread_local int creationStatus = QDMI_SUCCESS;
  createJob = library->device_session_create_device_job;
  const auto restore = llvm::make_scope_exit(
      [&] { library->device_session_create_device_job = createJob; });
  creations = 0;
  creationStatus = QDMI_SUCCESS;
  library->device_session_create_device_job = [](QDMI_Device_Session session,
                                                 QDMI_Device_Job* job) {
    ++creations;
    return creationStatus == QDMI_SUCCESS ? createJob(session, job)
                                          : creationStatus;
  };
  constexpr auto source =
      "OPENQASM 3.0; include \"stdgates.inc\"; "
      "qubit[2] q; bit[2] c; h q[0]; cx q[0],q[1]; c = measure q;";
  const auto compiled = ::mqt::test::value(
      mlir::compileProgram(mlir::OpenQASMProgram(source), device));
  EXPECT_EQ(creations, 0);
  EXPECT_TRUE(failed(mlir::submitProgram(device, compiled, -1)));
  EXPECT_EQ(::mqt::test::errorKind([&] {
              return mlir::submitProgram(device, mlir::OpenQASMProgram(source),
                                         -1);
            }),
            ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(creations, 0);
  EXPECT_FALSE(::mqt::test::errorMessage([&] {
                 return mlir::submitProgram(
                     device, mlir::OpenQASMProgram("invalid source"));
               }).empty());
  EXPECT_EQ(creations, 0);
  EXPECT_EQ(compiled.programFormat(), QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE);
  ASSERT_GE(compiled.payload().size(), 4);
  EXPECT_EQ(compiled.payload().substr(0, 2), "BC");
  creationStatus = QDMI_ERROR_PERMISSIONDENIED;
  const auto submissionError = ::mqt::test::diagnostic(
      [&] { return mlir::submitProgram(device, compiled, 32); });
  creationStatus = QDMI_SUCCESS;
  ASSERT_TRUE(submissionError);
  EXPECT_EQ(submissionError->status, QDMI_ERROR_PERMISSIONDENIED);
  EXPECT_EQ(submissionError->severity, ::mqt::DiagnosticSeverity::Error);
  EXPECT_TRUE(submissionError->message.starts_with(
      "Failed to submit compiled program: "));
  auto job = ::mqt::test::value(mlir::submitProgram(device, compiled, 32));
  EXPECT_EQ(creations, 2);
  EXPECT_TRUE(::mqt::test::value(job.wait()));
  EXPECT_EQ(::mqt::test::value(job.getShots()).size(), 32);
  const auto otherSession =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.ddsim.default"));
  auto otherJob =
      ::mqt::test::value(mlir::submitProgram(otherSession, compiled, 16));
  EXPECT_TRUE(::mqt::test::value(otherJob.wait()));
  EXPECT_EQ(::mqt::test::value(otherJob.getShots()).size(), 16);
  EXPECT_EQ(creations, 3);

  const auto explicitTarget = ::mqt::test::value(
      CompilerTarget::create(2, CompilerTarget::Connectivity::allToAll(),
                             CompilerTarget::NativeOperations::unrestricted()));
  const auto payload =
      ::mqt::test::value(mlir::payloadSpecificationForProgramFormat(
          QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE));
  const auto otherArtifact = ::mqt::test::value(mlir::CompiledProgram::compile(
      mlir::OpenQASMProgram(source),
      mlir::TargetEnvironment(explicitTarget, payload)));
  EXPECT_TRUE(mlir::failed(mlir::submitProgram(device, otherArtifact)));
  EXPECT_EQ(creations, 3);

  const auto restricted = ::mqt::test::value(mlir::PayloadSpecification::create(
      payload.format(), {{.id = "forward-branching"}}));
  const auto unsupported = ::mqt::test::errorMessage([&] {
    return mlir::CompiledProgram::compile(
        mlir::OpenQASMProgram(source),
        mlir::TargetEnvironment(compiled.environment().target(), restricted));
  });
  EXPECT_NE(unsupported.find("qir.dynamic-result-management"),
            std::string::npos);
  const auto future = ::mqt::test::value(
      mlir::PayloadSpecification::create({.id = "openqasm", .version = "3.2"}));
  EXPECT_EQ(
      ::mqt::test::errorKind([&] {
        return mlir::CompiledProgram::compile(
            mlir::OpenQASMProgram(source),
            mlir::TargetEnvironment(compiled.environment().target(), future));
      }),
      ::mqt::ErrorCategory::InvalidArgument);
  EXPECT_EQ(creations, 3);
}

TEST(CompilerQDMIAdapterTest, CompilesAdaptiveMeasurementControlledLoop) {
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.ddsim.default"));
  constexpr auto source =
      "OPENQASM 3.0; include \"stdgates.inc\"; qubit q; bit c; x q; "
      "c = measure q; while (c) { x q; c = measure q; }";
  auto compiled = ::mqt::test::value(
      mlir::compileProgram(mlir::OpenQASMProgram(source), device));
  auto job = ::mqt::test::value(mlir::submitProgram(device, compiled, 8));
  ASSERT_TRUE(::mqt::test::value(job.wait()));
  EXPECT_EQ(::mqt::test::value(job.getCounts()).at("0"), 8);
}

TEST(CompilerQDMIAdapterTest, ExecutesStableRegisterHelpers) {
  auto program = mlir::QCProgram::fromMLIRString(R"mlir(module {
    func.func private @move(%a: memref<1x!qc.qubit>, %b: memref<1x!qc.qubit>) {
      %zero = arith.constant 0 : index
      %left = memref.load %a[%zero] : memref<1x!qc.qubit>
      %right = memref.load %b[%zero] : memref<1x!qc.qubit>
      qc.x %left : !qc.qubit
      qc.swap %left, %right : !qc.qubit, !qc.qubit
      return
    }
    func.func @main() -> !cbit.reg<2> attributes {mqt.entry_point} {
      %a = memref.alloc() : memref<1x!qc.qubit>
      %b = memref.alloc() : memref<1x!qc.qubit>
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      func.call @move(%a, %b) : (memref<1x!qc.qubit>, memref<1x!qc.qubit>) -> ()
      %bits = cbit.alloc(#cbit.init<undefined>) : !cbit.reg<2>
      %q0 = memref.load %a[%zero] : memref<1x!qc.qubit>
      %q1 = memref.load %b[%zero] : memref<1x!qc.qubit>
      %m0 = qc.measure %q0 : !qc.qubit -> i1
      %m1 = qc.measure %q1 : !qc.qubit -> i1
      cbit.store %m0, %bits[%zero] : !cbit.reg<2>
      cbit.store %m1, %bits[%one] : !cbit.reg<2>
      memref.dealloc %a : memref<1x!qc.qubit>
      memref.dealloc %b : memref<1x!qc.qubit>
      return %bits : !cbit.reg<2>
    }
  })mlir");
  ASSERT_TRUE(program);
  ASSERT_TRUE(program->cleanup());
  auto qco = std::move(*program).intoQCO();
  ASSERT_TRUE(qco);
  program = std::move(*qco).intoQC();
  ASSERT_TRUE(program);
  auto qir = std::move(*program).intoQIR(mlir::QIRProfile::Adaptive);
  ASSERT_TRUE(qir);
  auto ir = qir->llvmIR();
  ASSERT_TRUE(ir);
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.ddsim.default"));
  auto job = ::mqt::test::value(
      device.submitJob(*ir, QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING, 8));
  ASSERT_TRUE(::mqt::test::value(job.wait()));
  EXPECT_EQ(::mqt::test::value(job.getCounts()).at("10"), 8);
}

TEST(CompilerQDMIAdapterTest, ValidatesCompiledEntryPointSignature) {
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.ddsim.default"));
  const auto environment = ::mqt::test::value(mlir::targetEnvironmentFromDevice(
      device, QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING));
  auto noResult = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() attributes {mqt.entry_point} { return }
    })");
  ASSERT_TRUE(noResult);
  const auto compiled = ::mqt::test::value(
      mlir::CompiledProgram::compile(std::move(*noResult), environment));
  EXPECT_NE(compiled.payload().find("define i64 @main()"), std::string::npos);

  auto narrowExitCode = mlir::QCProgram::fromMLIRString(R"(
    module {
      func.func @main() -> i32 attributes {mqt.entry_point} {
        %zero = arith.constant 0 : i32
        return %zero : i32
      }
    })");
  ASSERT_TRUE(narrowExitCode);
  const auto error = ::mqt::test::errorMessage([&] {
    return mlir::CompiledProgram::compile(std::move(*narrowExitCode),
                                          environment);
  });
  EXPECT_NE(error.find("requires an i64 () entry point"), std::string::npos)
      << error;
}

TEST(CompilerQDMIAdapterTest,
     CompatibilityPreservesOperandsWithinReorderedTuples) {
  const auto makeEnvironment =
      [](std::vector<std::vector<CompilerTarget::SiteId>> tuples) {
        std::vector<CompilerTarget::SiteTuple> siteTuples;
        siteTuples.reserve(tuples.size());
        for (auto& tuple : tuples) {
          siteTuples.push_back(::mqt::test::value(
              CompilerTarget::SiteTuple::create(std::move(tuple))));
        }
        const auto operation =
            ::mqt::test::value(CompilerTarget::OperationCapability::create(
                "cx", 2, 0, std::move(siteTuples)));
        const auto target = ::mqt::test::value(CompilerTarget::create(
            {
                ::mqt::test::value(CompilerTarget::Site::create(0)),
                ::mqt::test::value(CompilerTarget::Site::create(1)),
                ::mqt::test::value(CompilerTarget::Site::create(2)),
            },
            CompilerTarget::Connectivity::allToAll(),
            CompilerTarget::NativeOperations::fromOperations({operation})));
        return mlir::TargetEnvironment(
            target,
            ::mqt::test::value(mlir::payloadSpecificationForProgramFormat(
                QDMI_PROGRAM_FORMAT_QASM3)));
      };
  const auto original = makeEnvironment({{0, 1}, {1, 2}, {0, 2}});
  EXPECT_FALSE(failed(mlir::validateTargetCompatibility(
      original, makeEnvironment({{0, 2}, {1, 2}, {0, 1}}))));
  EXPECT_TRUE(failed(mlir::validateTargetCompatibility(
      original, makeEnvironment({{0, 2}, {2, 1}, {0, 1}}))));
}

TEST(CompilerQDMIAdapterTest, SubmissionQueriesOnlyTheRequiredMetadata) {
  const auto device =
      ::mqt::test::value(qdmi::Session::openDevice("mqt.ddsim.default"));
  auto* library = &const_cast<qdmi::DeviceLibrary&>(
      static_cast<QDMI_Device>(device)->getLibrary());
  static thread_local decltype(QDMI_device_session_query_device_property)*
      queryDevice = nullptr;
  static thread_local decltype(QDMI_device_session_query_site_property)*
      querySite = nullptr;
  static thread_local size_t siteLists = 0;
  static thread_local size_t calibrationQueries = 0;
  queryDevice = library->device_session_query_device_property;
  const auto restore = llvm::make_scope_exit([&] {
    library->device_session_query_device_property = queryDevice;
    library->device_session_query_site_property = querySite;
  });
  querySite = library->device_session_query_site_property;
  library->device_session_query_device_property =
      [](QDMI_Device_Session session, QDMI_Device_Property property,
         size_t size, void* value, size_t* sizeRet) {
        siteLists += property == QDMI_DEVICE_PROPERTY_SITES && value != nullptr;
        return queryDevice(session, property, size, value, sizeRet);
      };
  library->device_session_query_site_property =
      [](QDMI_Device_Session session, QDMI_Site site,
         QDMI_Site_Property property, size_t size, void* value,
         size_t* sizeRet) {
        calibrationQueries += property == QDMI_SITE_PROPERTY_NAME ||
                              property == QDMI_SITE_PROPERTY_T1 ||
                              property == QDMI_SITE_PROPERTY_T2;
        return querySite(session, site, property, size, value, sizeRet);
      };
  constexpr auto source = "OPENQASM 3.1; qubit q; bit c = measure q;";
  const auto compiled = ::mqt::test::value(
      mlir::compileProgram(mlir::OpenQASMProgram(source), device));
  siteLists = calibrationQueries = 0;
  auto job = ::mqt::test::value(mlir::submitProgram(device, compiled, 4));
  EXPECT_EQ(siteLists, 1);
  EXPECT_EQ(calibrationQueries, 0);
  ASSERT_TRUE(::mqt::test::value(job.wait()));
  EXPECT_EQ(::mqt::test::value(job.getCounts()).at("0"), 4);

  siteLists = calibrationQueries = 0;
  auto sourceJob = ::mqt::test::value(
      mlir::submitProgram(device, mlir::OpenQASMProgram(source), 4));
  EXPECT_EQ(siteLists, 1);
  EXPECT_GT(calibrationQueries, 0);
  ASSERT_TRUE(::mqt::test::value(sourceJob.wait()));
  EXPECT_EQ(::mqt::test::value(sourceJob.getCounts()).at("0"), 4);
}
