/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/QDMIAdapter.h"
#include "mlir/Compiler/Target.h"
#include "mlir/Compiler/TargetEnvironment.h"
#include "qdmi/Client.hpp"
#include "qdmi/ProgramFormat.hpp"
#include "qdmi/driver/Driver.hpp"

#include <gtest/gtest.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>
#include <qdmi/constants.h>
#include <qdmi/device.h>

#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using mlir::CompilerTarget;

class AdapterDeviceLibrary final : public qdmi::DeviceLibrary {
  static inline AdapterDeviceLibrary* activeLibrary = nullptr;

  [[nodiscard]] static AdapterDeviceLibrary&
  fromSession(QDMI_Device_Session session) {
    return *reinterpret_cast<AdapterDeviceLibrary*>(session);
  }

  static auto copyValue(const void* source, const size_t requiredSize,
                        const size_t size, void* value, size_t* sizeRet)
      -> int {
    if (value != nullptr && size < requiredSize) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (value != nullptr && requiredSize != 0U) {
      std::memcpy(value, source, requiredSize);
    }
    if (sizeRet != nullptr) {
      *sizeRet = requiredSize;
    }
    return QDMI_SUCCESS;
  }

  static auto alloc(QDMI_Device_Session* session) -> int {
    if (session == nullptr || activeLibrary == nullptr) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    *session = reinterpret_cast<QDMI_Device_Session>(activeLibrary);
    return QDMI_SUCCESS;
  }

  static auto init(QDMI_Device_Session session) -> int {
    return session == nullptr ? QDMI_ERROR_INVALIDARGUMENT : QDMI_SUCCESS;
  }

  static void free([[maybe_unused]] QDMI_Device_Session session) {}

  static auto queryDeviceProperty(QDMI_Device_Session session,
                                  const QDMI_Device_Property property,
                                  const size_t size, void* value,
                                  size_t* sizeRet) -> int {
    if (session == nullptr) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto& library = fromSession(session);
    switch (property) {
    case QDMI_DEVICE_PROPERTY_NAME: {
      static constexpr std::string_view DEVICE_NAME{"adapter-test"};
      return copyValue(DEVICE_NAME.data(), DEVICE_NAME.size() + 1U, size, value,
                       sizeRet);
    }
    case QDMI_DEVICE_PROPERTY_QUBITSNUM: {
      constexpr size_t numQubits = 1U;
      return copyValue(&numQubits, sizeof(numQubits), size, value, sizeRet);
    }
    case QDMI_DEVICE_PROPERTY_SITES: {
      auto* const site = reinterpret_cast<QDMI_Site>(&library);
      return copyValue(static_cast<const void*>(&site), sizeof(QDMI_Site), size,
                       value, sizeRet);
    }
    case QDMI_DEVICE_PROPERTY_COUPLINGMAP:
    case QDMI_DEVICE_PROPERTY_OPERATIONS:
      return copyValue(nullptr, 0U, size, value, sizeRet);
    case QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS:
      return copyValue(library.formats.data(),
                       library.formats.size() * sizeof(QDMI_Program_Format),
                       size, value, sizeRet);
    default:
      return QDMI_ERROR_NOTSUPPORTED;
    }
  }

  static auto
  queryProgramFeatures(QDMI_Device_Session session,
                       [[maybe_unused]] const QDMI_Program_Format* format,
                       const size_t size, QDMI_Program_Feature* value,
                       size_t* sizeRet) -> int {
    if (session == nullptr) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    const auto& features = fromSession(session).features;
    return copyValue(features.data(),
                     features.size() * sizeof(QDMI_Program_Feature), size,
                     value, sizeRet);
  }

  static auto querySiteProperty(QDMI_Device_Session session,
                                [[maybe_unused]] QDMI_Site site,
                                const QDMI_Site_Property property,
                                const size_t size, void* value, size_t* sizeRet)
      -> int {
    if (session == nullptr) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (property != QDMI_SITE_PROPERTY_INDEX) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    constexpr size_t index = 0U;
    return copyValue(&index, sizeof(index), size, value, sizeRet);
  }

public:
  std::vector<QDMI_Program_Format> formats{qdmi::OPENQASM3};
  std::vector<QDMI_Program_Feature> features;

  AdapterDeviceLibrary() {
    assert(activeLibrary == nullptr);
    activeLibrary = this;
    device_session_alloc = alloc;
    device_session_init = init;
    device_session_free = free;
    device_session_query_device_property = queryDeviceProperty;
    device_session_query_program_features = queryProgramFeatures;
    device_session_query_site_property = querySiteProperty;
  }

  ~AdapterDeviceLibrary() override {
    assert(activeLibrary == this);
    activeLibrary = nullptr;
  }
};

class CompilerQDMIPayloadAdapterTest : public testing::Test {
protected:
  std::shared_ptr<AdapterDeviceLibrary> library_ =
      std::make_shared<AdapterDeviceLibrary>();
  QDMI_Device_impl_d handle_{library_};
  qdmi::Device device_ = qdmi::Session::createSessionlessDevice(&handle_);
};

} // namespace

[[nodiscard]] static const mlir::ProgramCapability&
findCapability(const mlir::PayloadSpecification& payload,
               const llvm::StringRef id) {
  const auto* const found =
      llvm::find_if(payload.capabilities(), [&](const auto& capability) {
        return capability.id == id;
      });
  assert(found != payload.capabilities().end() &&
         "Payload capability not found");
  return *found;
}

[[nodiscard]] static const CompilerTarget::Operation&
findOperation(const CompilerTarget& target, const llvm::StringRef name) {
  const auto* const found =
      llvm::find_if(target.operations(),
                    [&](const auto& op) { return op.canonicalName() == name; });
  assert(found != target.operations().end() && "Target operation not found");
  return *found;
}

TEST(CompilerQDMIAdapterTest, SnapshotsIQMCalibrationAndLifetime) {
  const auto target = llvm::cantFail([] {
    const auto device = qdmi::Session::openDevice("mqt.sc.iqm.garnet");
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

TEST(CompilerQDMIAdapterTest, InfersDDSIMTargetFacts) {
  const auto device = qdmi::Session::openDevice("mqt.ddsim.default");
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

  EXPECT_EQ(target.numSites(), 65535);
  EXPECT_EQ(target.connectivityKind(),
            CompilerTarget::Connectivity::Kind::AllToAll);
  EXPECT_EQ(target.nativeOperationsKind(),
            CompilerTarget::NativeOperations::Kind::Explicit);
  const auto& gphase = findOperation(target, "gphase");
  EXPECT_EQ(gphase.arity().kind(),
            CompilerTarget::Operation::Arity::Kind::Fixed);
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
              CompilerTarget::Operation::Arity::Kind::Variadic)
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

TEST(CompilerQDMIAdapterTest, SnapshotsExactPayloadAndFeatureGroups) {
  const auto device = qdmi::Session::openDevice("mqt.ddsim.default");
  const auto environment = llvm::cantFail(
      mlir::targetEnvironmentFromDevice(device, qdmi::OPENQASM3));
  const auto& payload = environment.payloadSpecification();

  EXPECT_EQ(payload.format(),
            (mlir::PayloadFormat{.id = "openqasm",
                                 .version = "3.0.0",
                                 .profile = "",
                                 .encoding = mlir::PayloadEncoding::Text}));
  EXPECT_TRUE(payload.optionalCapabilitiesKnown());
  ASSERT_EQ(payload.capabilities().size(), 5);
  for (const llvm::StringRef id : {
           QDMI_PROGRAM_FEATURE_MID_CIRCUIT_MEASUREMENT,
           QDMI_PROGRAM_FEATURE_MEASURED_QUBIT_REUSE,
           QDMI_PROGRAM_FEATURE_MEASUREMENT_RESULT_USE,
           QDMI_PROGRAM_FEATURE_BOOLEAN_COMPUTATION,
           QDMI_PROGRAM_FEATURE_FORWARD_BRANCHING,
       }) {
    const auto& capability = findCapability(payload, id);
    EXPECT_EQ(capability.value, 0);
    EXPECT_TRUE(capability.constraints.empty());
  }
}

TEST(CompilerQDMIAdapterTest, AddsQIRAdaptiveBaselineOnce) {
  const auto environment = llvm::cantFail(mlir::targetEnvironmentFromDeviceId(
      "mqt.ddsim.default", qdmi::QIR21_ADAPTIVE_BINARY));
  const auto& payload = environment.payloadSpecification();

  EXPECT_EQ(payload.format(),
            (mlir::PayloadFormat{.id = "qir",
                                 .version = "2.1.0",
                                 .profile = "adaptive",
                                 .encoding = mlir::PayloadEncoding::Binary}));
  EXPECT_TRUE(payload.optionalCapabilitiesKnown());
  ASSERT_EQ(payload.capabilities().size(), 5);
  for (const llvm::StringRef id : {
           QDMI_PROGRAM_FEATURE_MID_CIRCUIT_MEASUREMENT,
           QDMI_PROGRAM_FEATURE_MEASURED_QUBIT_REUSE,
           QDMI_PROGRAM_FEATURE_MEASUREMENT_RESULT_USE,
           QDMI_PROGRAM_FEATURE_BOOLEAN_COMPUTATION,
           QDMI_PROGRAM_FEATURE_FORWARD_BRANCHING,
       }) {
    EXPECT_TRUE(findCapability(payload, id).constraints.empty());
  }
}

TEST_F(CompilerQDMIPayloadAdapterTest, GroupsConstrainedFeatures) {
  library_->features = {
      {
          .id = QDMI_PROGRAM_FEATURE_COUNTED_ITERATION,
          .value = 0U,
          .constraint_id =
              QDMI_PROGRAM_CONSTRAINT_MAX_CONTROL_FLOW_NESTING_DEPTH,
          .constraint_value = 3U,
      },
      {
          .id = QDMI_PROGRAM_FEATURE_COUNTED_ITERATION,
          .value = 0U,
          .constraint_id = QDMI_PROGRAM_CONSTRAINT_MAX_ITERATION_COUNT,
          .constraint_value = 100U,
      },
  };

  const auto environment = llvm::cantFail(
      mlir::targetEnvironmentFromDevice(device_, qdmi::OPENQASM3));
  const auto& capability =
      findCapability(environment.payloadSpecification(),
                     QDMI_PROGRAM_FEATURE_COUNTED_ITERATION);

  EXPECT_EQ(capability.value, 0U);
  EXPECT_EQ(
      capability.constraints,
      (std::vector<mlir::ProgramConstraint>{
          {.id = QDMI_PROGRAM_CONSTRAINT_MAX_CONTROL_FLOW_NESTING_DEPTH,
           .value = 3U},
          {.id = QDMI_PROGRAM_CONSTRAINT_MAX_ITERATION_COUNT, .value = 100U}}));
}

TEST_F(CompilerQDMIPayloadAdapterTest, RejectsInvalidFeatureGroups) {
  const auto expectError = [&](std::vector<QDMI_Program_Feature> features) {
    library_->features = std::move(features);
    auto environment =
        mlir::targetEnvironmentFromDevice(device_, qdmi::OPENQASM3);
    ASSERT_FALSE(environment);
    llvm::consumeError(environment.takeError());
  };

  expectError({
      QDMI_PROGRAM_FEATURE_UNCONSTRAINED(QDMI_PROGRAM_FEATURE_COUNTED_ITERATION,
                                         0U),
      QDMI_PROGRAM_FEATURE_UNCONSTRAINED(QDMI_PROGRAM_FEATURE_COUNTED_ITERATION,
                                         0U),
  });
  expectError({
      QDMI_PROGRAM_FEATURE_UNCONSTRAINED(QDMI_PROGRAM_FEATURE_COUNTED_ITERATION,
                                         0U),
      {
          .id = QDMI_PROGRAM_FEATURE_COUNTED_ITERATION,
          .value = 0U,
          .constraint_id = QDMI_PROGRAM_CONSTRAINT_MAX_ITERATION_COUNT,
          .constraint_value = 10U,
      },
  });
  expectError({
      {
          .id = QDMI_PROGRAM_FEATURE_COUNTED_ITERATION,
          .value = 0U,
          .constraint_id = QDMI_PROGRAM_CONSTRAINT_MAX_ITERATION_COUNT,
          .constraint_value = 10U,
      },
      {
          .id = QDMI_PROGRAM_FEATURE_COUNTED_ITERATION,
          .value = 0U,
          .constraint_id = QDMI_PROGRAM_CONSTRAINT_MAX_ITERATION_COUNT,
          .constraint_value = 20U,
      },
  });
}

TEST_F(CompilerQDMIPayloadAdapterTest, RejectsInvalidProgramFormats) {
  auto missingVersion = qdmi::OPENQASM3;
  missingVersion.version = 0U;
  auto invalidEncoding = qdmi::OPENQASM3;
  invalidEncoding.encoding = 3U;

  for (const auto& format : {missingVersion, invalidEncoding}) {
    auto environment = mlir::targetEnvironmentFromDevice(device_, format);
    ASSERT_FALSE(environment);
    EXPECT_NE(llvm::toString(environment.takeError()).find("not canonical"),
              std::string::npos);
  }
}

TEST(CompilerQDMIAdapterTest, RejectsPayloadNotAcceptedByDevice) {
  constexpr QDMI_Program_Format unsupported{
      .version = QDMI_MAKE_VERSION(3, 1, 0),
      .encoding = QDMI_PROGRAM_ENCODING_TEXT,
      .id = "openqasm",
      .profile = "",
  };
  const auto device = qdmi::Session::openDevice("mqt.ddsim.default");
  auto environment = mlir::targetEnvironmentFromDevice(device, unsupported);

  ASSERT_FALSE(environment);
  EXPECT_NE(llvm::toString(environment.takeError()).find("does not accept"),
            std::string::npos);
}

TEST(CompilerQDMIAdapterTest, ListsRegisteredDeviceIds) {
  const auto deviceIds = llvm::cantFail(mlir::registeredQDMIDeviceIds());
  EXPECT_TRUE(llvm::is_contained(deviceIds, "mqt.ddsim.default"));
}

TEST(CompilerQDMIAdapterTest, ConvertsUnknownDeviceFailureToError) {
  auto target = mlir::compilerTargetFromDeviceId("mqt.unknown.device");
  ASSERT_FALSE(target);
  const auto message = llvm::toString(target.takeError());
  EXPECT_NE(message.find("mqt.unknown.device"), std::string::npos);
  EXPECT_NE(message.find("Unknown QDMI device ID"), std::string::npos);

  auto environment = mlir::targetEnvironmentFromDeviceId("mqt.unknown.device",
                                                         qdmi::OPENQASM3);
  ASSERT_FALSE(environment);
  const auto environmentMessage = llvm::toString(environment.takeError());
  EXPECT_NE(environmentMessage.find("mqt.unknown.device"), std::string::npos);
  EXPECT_NE(environmentMessage.find("Unknown QDMI device ID"),
            std::string::npos);
}

TEST(CompilerQDMIAdapterTest, RejectsNonhomogeneousOperationSupport) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration =
      qdmi::FileDeviceConfiguration{MQT_CORE_MLIR_HETEROGENEOUS_SC_CONFIG};
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  auto environment = mlir::targetEnvironmentFromDevice(device, qdmi::OPENQASM3);
  ASSERT_FALSE(environment);
  const auto message = llvm::toString(environment.takeError());
  EXPECT_NE(message.find("homogeneous"), std::string::npos);
  EXPECT_NE(message.find("all topology edges"), std::string::npos);
}

TEST(CompilerQDMIAdapterTest, SnapshotsHomogeneousHigherArityOperation) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration =
      qdmi::FileDeviceConfiguration{MQT_CORE_MLIR_HIGHER_ARITY_SC_CONFIG};
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0));
  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0, {0, 1, 2}));
  EXPECT_TRUE(target.supportsOperation("ccnot", 3, 0, {2, 1, 0}));
  EXPECT_FALSE(target.supportsOperation("ccnot", 3, 0, {0, 1, 3}));
}

TEST(CompilerQDMIAdapterTest, PreservesOneWayDirectionalOperationSupport) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration = qdmi::FileDeviceConfiguration{
      MQT_CORE_MLIR_DIRECTIONAL_ONE_WAY_SC_CONFIG,
  };
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

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
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration = qdmi::InlineDeviceConfiguration{
      .json = R"({
    "schema-version": 1,
    "name": "Unavailable operation",
    "numQubits": 1,
    "durationUnit": {"unit": "ns", "scaleFactor": 1},
    "qubitProperties": {"defaults": {}, "overrides": []},
    "couplings": [],
    "operations": [
      {"name": "x", "numQubits": 1, "numParameters": 0, "sites": []}
    ]
  })",
  };
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));
  EXPECT_EQ(target.nativeOperationsKind(),
            CompilerTarget::NativeOperations::Kind::Explicit);
  EXPECT_TRUE(target.operations().empty());
  EXPECT_FALSE(target.supportsOperation("x", 1, 0, {0}));
}

TEST(CompilerQDMIAdapterTest,
     PreservesDirectionalCalibrationWhenBothOrientationsExist) {
  qdmi::DeviceSessionConfig overrides;
  overrides.deviceConfiguration = qdmi::FileDeviceConfiguration{
      MQT_CORE_MLIR_DIRECTIONAL_TWO_WAY_SC_CONFIG,
  };
  const auto device = qdmi::Session::openDevice("mqt.sc.default", overrides);
  const auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));

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
