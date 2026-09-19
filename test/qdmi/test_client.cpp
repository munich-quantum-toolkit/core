/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Client.hpp"
#include "qdmi/common/Common.hpp"

#include "TestUtils.hpp"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include "qdmi/client.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <variant>
#include <vector>

namespace qdmi {

namespace {

auto queryBytes(const std::vector<std::byte>& bytes) {
  return [&bytes](const size_t size, void* value, size_t* sizeRet) {
    if (sizeRet != nullptr) {
      *sizeRet = bytes.size();
    }
    if (value != nullptr) {
      if (size < bytes.size()) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      std::memcpy(value, bytes.data(), bytes.size());
    }
    return QDMI_SUCCESS;
  };
}

template <typename T> auto bytesOf(const T& value) {
  std::vector<std::byte> bytes(sizeof(T));
  std::memcpy(bytes.data(), &value, sizeof(T));
  return bytes;
}

class DeviceTest : public testing::TestWithParam<Device> {
protected:
  Device device;

  DeviceTest() : device(GetParam()) {}
};

class SiteTest : public DeviceTest {
protected:
  std::vector<Site> sites;

  void SetUp() override { sites = mqt::test::value(device.getSites()); }
};

class OperationTest : public DeviceTest {
protected:
  std::vector<Operation> operations;

  void SetUp() override {
    operations = mqt::test::value(device.getOperations());
  }
};

#ifdef MQT_CORE_QDMI_HAS_DDSIM_DEVICE
class DDSimulatorDeviceTest : public testing::Test {
protected:
  Device device;

  DDSimulatorDeviceTest() : device(getDDSimulatorDevice()) {}

private:
  static auto getDDSimulatorDevice() -> Device {
    auto session = mqt::test::value(qdmi::Session::create());
    for (const auto& dev : mqt::test::value(session.getDevices())) {
      if (mqt::test::value(dev.getName()) == "MQT Core DDSIM QDMI Device") {
        return dev;
      }
    }
    throw std::runtime_error("DD simulator device not found");
  }
};

class JobTest : public DDSimulatorDeviceTest {
protected:
  Job job;

  JobTest() : job(createTestJob()) {}

  [[nodiscard]] Job createTestJob() const {
    const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
h q[0];
c[0] = measure q[0];
)";
    return mqt::test::value(
        device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 10));
  }
};

class SimulatorJobTest : public DDSimulatorDeviceTest {
protected:
  Job job;

  SimulatorJobTest() : job(createTestJob()) {}

  [[nodiscard]] Job createTestJob() const {
    const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[2] q;
h q[0];
cx q[0], q[1];
)";
    return mqt::test::value(
        device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 0));
  }
};
#endif

} // namespace

TEST(CustomPropertyTest, SelectorsMapToEveryQDMIPropertyFamily) {
  constexpr std::array properties{
      CustomProperty::Custom1, CustomProperty::Custom2, CustomProperty::Custom3,
      CustomProperty::Custom4, CustomProperty::Custom5,
  };
  constexpr std::array deviceProperties{
      QDMI_DEVICE_PROPERTY_CUSTOM1, QDMI_DEVICE_PROPERTY_CUSTOM2,
      QDMI_DEVICE_PROPERTY_CUSTOM3, QDMI_DEVICE_PROPERTY_CUSTOM4,
      QDMI_DEVICE_PROPERTY_CUSTOM5,
  };
  constexpr std::array siteProperties{
      QDMI_SITE_PROPERTY_CUSTOM1, QDMI_SITE_PROPERTY_CUSTOM2,
      QDMI_SITE_PROPERTY_CUSTOM3, QDMI_SITE_PROPERTY_CUSTOM4,
      QDMI_SITE_PROPERTY_CUSTOM5,
  };
  constexpr std::array operationProperties{
      QDMI_OPERATION_PROPERTY_CUSTOM1, QDMI_OPERATION_PROPERTY_CUSTOM2,
      QDMI_OPERATION_PROPERTY_CUSTOM3, QDMI_OPERATION_PROPERTY_CUSTOM4,
      QDMI_OPERATION_PROPERTY_CUSTOM5,
  };
  constexpr std::array jobProperties{
      QDMI_JOB_PROPERTY_CUSTOM1, QDMI_JOB_PROPERTY_CUSTOM2,
      QDMI_JOB_PROPERTY_CUSTOM3, QDMI_JOB_PROPERTY_CUSTOM4,
      QDMI_JOB_PROPERTY_CUSTOM5,
  };
  constexpr std::array jobResults{
      QDMI_JOB_RESULT_CUSTOM1, QDMI_JOB_RESULT_CUSTOM2, QDMI_JOB_RESULT_CUSTOM3,
      QDMI_JOB_RESULT_CUSTOM4, QDMI_JOB_RESULT_CUSTOM5,
  };

  for (size_t i = 0; i < properties.size(); ++i) {
    EXPECT_EQ(mqt::test::value(detail::toDeviceProperty(properties[i])),
              deviceProperties[i]);
    EXPECT_EQ(mqt::test::value(detail::toSiteProperty(properties[i])),
              siteProperties[i]);
    EXPECT_EQ(mqt::test::value(detail::toOperationProperty(properties[i])),
              operationProperties[i]);
    EXPECT_EQ(mqt::test::value(detail::toJobProperty(properties[i])),
              jobProperties[i]);
    EXPECT_EQ(mqt::test::value(detail::toJobResult(properties[i])),
              jobResults[i]);
  }
}

TEST(CustomPropertyTest, RejectsInvalidSelector) {
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalid = static_cast<CustomProperty>(0);
  EXPECT_EQ(mqt::test::errorStatus(detail::toDeviceProperty(invalid)),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(mqt::test::errorStatus(detail::toSiteProperty(invalid)),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(mqt::test::errorStatus(detail::toOperationProperty(invalid)),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(mqt::test::errorStatus(detail::toJobProperty(invalid)),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(mqt::test::errorStatus(detail::toJobResult(invalid)),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(StandardPropertyTest, PreservesValuesAndOptionalSupport) {
  const auto bytes = bytesOf(size_t{42});
  const auto query = queryBytes(bytes);
  EXPECT_EQ(
      mqt::test::value(detail::queryProperty<size_t>(query, "value", "size")),
      42);
  EXPECT_EQ(mqt::test::value(detail::queryProperty<std::optional<size_t>>(
                query, "value", "size")),
            42);
  EXPECT_EQ(mqt::test::value(detail::queryProperty<std::vector<size_t>>(
                query, "value", "size")),
            std::vector<size_t>{42});
  const std::vector<std::byte> text{std::byte{'x'}, std::byte{0}};
  EXPECT_EQ(mqt::test::value(detail::queryProperty<std::string>(
                queryBytes(text), "value", "size")),
            "x");

  const auto unsupported = [](size_t, void*, size_t*) {
    return QDMI_ERROR_NOTSUPPORTED;
  };
  EXPECT_EQ(mqt::test::value(detail::queryProperty<std::optional<size_t>>(
                unsupported, "value", "size")),
            std::nullopt);
  EXPECT_EQ(mqt::test::value(detail::queryProperty<std::optional<std::string>>(
                unsupported, "value", "size")),
            std::nullopt);
  EXPECT_EQ(mqt::test::value(
                detail::queryProperty<std::optional<std::vector<size_t>>>(
                    unsupported, "value", "size")),
            std::nullopt);
  EXPECT_TRUE(mqt::test::errorStatus(
                  detail::queryProperty<size_t>(unsupported, "value", "size"))
                  .has_value());
}

TEST(StandardPropertyTest, ReturnsValuesUnsupportedPropertiesAndDiagnostics) {
  const auto bytes = bytesOf(size_t{42});
  auto value =
      detail::queryProperty<size_t>(queryBytes(bytes), "value", "size");
  ASSERT_TRUE(std::holds_alternative<size_t>(value));
  EXPECT_EQ(std::get<size_t>(value), 42);

  const auto unsupported = [](size_t, void*, size_t*) {
    return QDMI_ERROR_NOTSUPPORTED;
  };
  auto optional = detail::queryProperty<std::optional<size_t>>(unsupported,
                                                               "value", "size");
  ASSERT_TRUE(std::holds_alternative<std::optional<size_t>>(optional));
  EXPECT_FALSE(std::get<std::optional<size_t>>(optional));
  value = detail::queryProperty<size_t>(unsupported, "value", "size");
  ASSERT_TRUE(std::holds_alternative<Error>(value));
  EXPECT_EQ(std::get<Error>(value).status, QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(std::get<Error>(value).message, "value: Not supported.");

  const auto failsToRead = [](size_t, void* output, size_t* sizeRet) {
    if (sizeRet != nullptr) {
      *sizeRet = 4;
    }
    return output == nullptr ? QDMI_SUCCESS : QDMI_ERROR_BADSTATE;
  };
  auto text = detail::queryProperty<std::string>(failsToRead, "value", "size");
  ASSERT_TRUE(std::holds_alternative<Error>(text));
  EXPECT_EQ(std::get<Error>(text).message, "value: Bad state.");
  const std::vector<std::byte> unterminated{std::byte{'x'}};
  text = detail::queryProperty<std::string>(queryBytes(unterminated), "value",
                                            "size");
  ASSERT_TRUE(std::holds_alternative<Error>(text));
  EXPECT_EQ(std::get<Error>(text).message, "value: missing string terminator");
}

TEST(StandardPropertyTest, RejectsMalformedSizesBeforeReading) {
  bool read = false;
  const auto query = [&read](size_t, void* value, size_t* sizeRet) {
    if (sizeRet != nullptr) {
      *sizeRet = sizeof(size_t) + 1;
    }
    read |= value != nullptr;
    return QDMI_SUCCESS;
  };
  EXPECT_TRUE(mqt::test::errorStatus(detail::queryProperty<std::vector<size_t>>(
                                         query, "value", "size"))
                  .has_value());
  EXPECT_TRUE(mqt::test::errorStatus(
                  detail::queryProperty<std::optional<std::vector<size_t>>>(
                      query, "value", "size"))
                  .has_value());
  EXPECT_FALSE(read);
}

TEST(StandardPropertyTest, ValidatesStringsAndPreservesEmptyValues) {
  const std::vector<std::byte> empty;
  EXPECT_TRUE(mqt::test::value(detail::queryProperty<std::vector<size_t>>(
                                   queryBytes(empty), "value", "size"))
                  .empty());
  EXPECT_TRUE(mqt::test::errorStatus(detail::queryProperty<std::string>(
                                         queryBytes(empty), "value", "size"))
                  .has_value());
  const std::vector<std::byte> unterminated{std::byte{'x'}};
  EXPECT_TRUE(
      mqt::test::errorStatus(detail::queryProperty<std::optional<std::string>>(
                                 queryBytes(unterminated), "value", "size"))
          .has_value());
  const std::vector<std::byte> terminated{std::byte{0}};
  EXPECT_EQ(mqt::test::value(detail::queryProperty<std::string>(
                queryBytes(terminated), "value", "size")),
            "");
}

TEST(CustomPropertyTest, DecodesSupportedTypes) {
  const std::vector<std::byte> stringBytes{
      std::byte{'v'}, std::byte{'a'}, std::byte{'l'},
      std::byte{'u'}, std::byte{'e'}, std::byte{0},
  };
  EXPECT_EQ(mqt::test::value(detail::queryCustomValue<std::string>(
                queryBytes(stringBytes), "test property")),
            "value");

  constexpr bool boolValue = true;
  EXPECT_EQ(mqt::test::value(detail::queryCustomValue<bool>(
                queryBytes(bytesOf(boolValue)), "test property")),
            boolValue);
  constexpr int intValue = 42;
  EXPECT_EQ(mqt::test::value(detail::queryCustomValue<int>(
                queryBytes(bytesOf(intValue)), "test property")),
            intValue);
  constexpr double doubleValue = 1.25;
  EXPECT_EQ(mqt::test::value(detail::queryCustomValue<double>(
                queryBytes(bytesOf(doubleValue)), "test property")),
            doubleValue);
  EXPECT_EQ(mqt::test::value(detail::queryCustomValue<std::vector<std::byte>>(
                queryBytes(stringBytes), "test property")),
            stringBytes);
}

TEST(CustomPropertyTest, ReturnsNulloptWhenUnsupported) {
  const auto query = [](size_t, void*, size_t*) {
    return QDMI_ERROR_NOTSUPPORTED;
  };
  EXPECT_EQ(
      mqt::test::value(detail::queryCustomValue<int>(query, "test property")),
      std::nullopt);
}

TEST(CustomPropertyTest, PropagatesQueryErrors) {
  const auto failingSizeQuery = [](size_t, void*, size_t*) {
    return QDMI_ERROR_INVALIDARGUMENT;
  };
  EXPECT_EQ(mqt::test::errorStatus(detail::queryCustomValue<int>(
                failingSizeQuery, "test property")),
            QDMI_ERROR_INVALIDARGUMENT);

  const auto failingValueQuery = [](const size_t, void* value,
                                    size_t* sizeRet) {
    if (sizeRet != nullptr) {
      *sizeRet = sizeof(int);
      return QDMI_SUCCESS;
    }
    EXPECT_NE(value, nullptr);
    return QDMI_ERROR_INVALIDARGUMENT;
  };
  EXPECT_EQ(mqt::test::errorStatus(detail::queryCustomValue<int>(
                failingValueQuery, "test property")),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(CustomPropertyTest, SupportsEmptyRawValues) {
  const std::vector<std::byte> empty;
  EXPECT_EQ(mqt::test::value(detail::queryCustomValue<std::vector<std::byte>>(
                queryBytes(empty), "test property")),
            empty);
}

TEST(CustomPropertyTest, RejectsIncompatibleRepresentations) {
  const std::vector<std::byte> empty;
  EXPECT_EQ(mqt::test::errorStatus(detail::queryCustomValue<std::string>(
                queryBytes(empty), "test property")),
            QDMI_ERROR_INVALIDARGUMENT);
  const std::vector<std::byte> malformedString{std::byte{'n'}, std::byte{'o'}};
  EXPECT_EQ(mqt::test::errorStatus(detail::queryCustomValue<std::string>(
                queryBytes(malformedString), "test property")),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(mqt::test::errorStatus(detail::queryCustomValue<double>(
                queryBytes(bytesOf(true)), "test property")),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(QueuePositionTest, ReturnsPositionOnSuccess) {
  EXPECT_EQ(mqt::test::value(detail::queuePositionFromResult(QDMI_SUCCESS, 3)),
            3);
}

TEST(QueuePositionTest, ReturnsNulloptWhenUnavailable) {
  EXPECT_EQ(mqt::test::value(
                detail::queuePositionFromResult(QDMI_ERROR_NOTSUPPORTED, 0)),
            std::nullopt);
  EXPECT_EQ(
      mqt::test::value(detail::queuePositionFromResult(QDMI_ERROR_BADSTATE, 0)),
      std::nullopt);
}

TEST(QueuePositionTest, PropagatesOtherQueryErrors) {
  EXPECT_EQ(mqt::test::errorStatus(
                detail::queuePositionFromResult(QDMI_ERROR_INVALIDARGUMENT, 0)),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST(JobShotsTest, PreservesZeroWidthShots) {
  EXPECT_EQ(mqt::test::value(detail::parseShots("", 0)),
            std::vector<std::string>{});
  EXPECT_EQ(mqt::test::value(detail::parseShots("", 1)),
            std::vector<std::string>{""});
  EXPECT_EQ(mqt::test::value(detail::parseShots(",,,", 4)),
            (std::vector<std::string>{"", "", "", ""}));
}

TEST(JobShotsTest, ValidatesShotCount) {
  EXPECT_TRUE(mqt::test::errorStatus(detail::parseShots("0,1", 1)).has_value());
  EXPECT_TRUE(mqt::test::errorStatus(detail::parseShots("0", 2)).has_value());
}

TEST(QDMITest, StatusToString) {
  EXPECT_STREQ(qdmi::toString(QDMI_WARN_GENERAL), "General warning");
  EXPECT_STREQ(qdmi::toString(QDMI_SUCCESS), "Success");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_FATAL), "A fatal error");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_OUTOFMEM), "Out of memory");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_NOTIMPLEMENTED), "Not implemented");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_LIBNOTFOUND), "Library not found");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_NOTFOUND), "Element not found");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_OUTOFRANGE), "Out of range");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_INVALIDARGUMENT), "Invalid argument");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_PERMISSIONDENIED),
               "Permission denied");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_NOTSUPPORTED), "Not supported");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_BADSTATE), "Bad state");
  EXPECT_STREQ(qdmi::toString(QDMI_ERROR_TIMEOUT), "Timeout");
}

TEST(QDMITest, SitePropertyToString) {
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_INDEX), "INDEX");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_T1), "T1");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_T2), "T2");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_NAME), "NAME");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_XCOORDINATE), "X COORDINATE");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_YCOORDINATE), "Y COORDINATE");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_ZCOORDINATE), "Z COORDINATE");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_ISZONE), "IS ZONE");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_XEXTENT), "X EXTENT");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_YEXTENT), "Y EXTENT");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_ZEXTENT), "Z EXTENT");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_MODULEINDEX), "MODULE INDEX");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_SUBMODULEINDEX),
               "SUBMODULE INDEX");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_MAX), "MAX");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_CUSTOM1), "CUSTOM1");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_CUSTOM2), "CUSTOM2");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_CUSTOM3), "CUSTOM3");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_CUSTOM4), "CUSTOM4");
  EXPECT_STREQ(qdmi::toString(QDMI_SITE_PROPERTY_CUSTOM5), "CUSTOM5");
}

TEST(QDMITest, OperationPropertyToString) {
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_NAME), "NAME");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_QUBITSNUM), "QUBITS NUM");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_PARAMETERSNUM),
               "PARAMETERS NUM");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_DURATION), "DURATION");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_FIDELITY), "FIDELITY");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_INTERACTIONRADIUS),
               "INTERACTION RADIUS");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_BLOCKINGRADIUS),
               "BLOCKING RADIUS");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_IDLINGFIDELITY),
               "IDLING FIDELITY");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_ISZONED), "IS ZONED");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_MEANSHUTTLINGSPEED),
               "MEAN SHUTTLING SPEED");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_MAX), "MAX");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_CUSTOM1), "CUSTOM1");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_CUSTOM2), "CUSTOM2");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_CUSTOM3), "CUSTOM3");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_CUSTOM4), "CUSTOM4");
  EXPECT_STREQ(qdmi::toString(QDMI_OPERATION_PROPERTY_CUSTOM5), "CUSTOM5");
}

TEST(QDMITest, DevicePropertyToString) {
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_NAME), "NAME");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_VERSION), "VERSION");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_STATUS), "STATUS");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_LIBRARYVERSION),
               "LIBRARY VERSION");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_QUBITSNUM), "QUBITS NUM");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_SITES), "SITES");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_OPERATIONS), "OPERATIONS");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_COUPLINGMAP),
               "COUPLING MAP");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION),
               "NEEDS CALIBRATION");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_LENGTHUNIT), "LENGTH UNIT");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_LENGTHSCALEFACTOR),
               "LENGTH SCALE FACTOR");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_DURATIONUNIT),
               "DURATION UNIT");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_DURATIONSCALEFACTOR),
               "DURATION SCALE FACTOR");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_MINATOMDISTANCE),
               "MIN ATOM DISTANCE");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS),
               "SUPPORTED PROGRAM FORMATS");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_CHILDDEVICES),
               "CHILD DEVICES");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_QUEUELENGTH),
               "QUEUE LENGTH");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_MAX), "MAX");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_CUSTOM1), "CUSTOM1");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_CUSTOM2), "CUSTOM2");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_CUSTOM3), "CUSTOM3");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_CUSTOM4), "CUSTOM4");
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_PROPERTY_CUSTOM5), "CUSTOM5");
}

TEST(QDMITest, SessionPropertyToString) {
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PROPERTY_DEVICES), "DEVICES");
}

TEST(QDMITest, DeviceSessionParameterToString) {
  EXPECT_STREQ(qdmi::toString(QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE),
               "CHILD DEVICE");
}

TEST(QDMITest, CheckErrorPreservesStatuses) {
  EXPECT_FALSE(qdmi::checkError(QDMI_SUCCESS, "Test"));
  EXPECT_FALSE(qdmi::checkError(QDMI_WARN_GENERAL, "Test"));
  for (int code = QDMI_ERROR_TIMEOUT; code <= QDMI_ERROR_FATAL; ++code) {
    const auto error = qdmi::checkError(code, "Test");
    ASSERT_TRUE(error);
    EXPECT_EQ(error->status, code);
    EXPECT_THAT(error->message, testing::HasSubstr("Test"));
  }
  EXPECT_EQ(qdmi::checkError(-99, "Test")->status, -99);
}

TEST(QDMITest, BinaryProgramFormatClassification) {
  // The switch below states the expected classification of every program
  // format. It has no default case, so a format added to QDMI later produces
  // an unhandled-enumerator warning instead of an unnoticed classification.
  constexpr auto expected = [](const QDMI_Program_Format format) -> bool {
    switch (format) {
    case QDMI_PROGRAM_FORMAT_QIRBASEMODULE:
    case QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE:
    case QDMI_PROGRAM_FORMAT_QPY:
      return true;
    case QDMI_PROGRAM_FORMAT_QASM2:
    case QDMI_PROGRAM_FORMAT_QASM3:
    case QDMI_PROGRAM_FORMAT_QIRBASESTRING:
    case QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING:
    case QDMI_PROGRAM_FORMAT_CALIBRATION:
    case QDMI_PROGRAM_FORMAT_IQMJSON:
    case QDMI_PROGRAM_FORMAT_BATCHJOB:
    case QDMI_PROGRAM_FORMAT_CUSTOM1:
    case QDMI_PROGRAM_FORMAT_CUSTOM2:
    case QDMI_PROGRAM_FORMAT_CUSTOM3:
    case QDMI_PROGRAM_FORMAT_CUSTOM4:
    case QDMI_PROGRAM_FORMAT_CUSTOM5:
      return false;
    }
    return false;
  };

  // Every program format QDMI defines. A format added to QDMI must be added
  // here as well so that the loop below covers it.
  constexpr std::array formats{
      QDMI_PROGRAM_FORMAT_QASM2,
      QDMI_PROGRAM_FORMAT_QASM3,
      QDMI_PROGRAM_FORMAT_QIRBASESTRING,
      QDMI_PROGRAM_FORMAT_QIRBASEMODULE,
      QDMI_PROGRAM_FORMAT_QIRADAPTIVESTRING,
      QDMI_PROGRAM_FORMAT_QIRADAPTIVEMODULE,
      QDMI_PROGRAM_FORMAT_CALIBRATION,
      QDMI_PROGRAM_FORMAT_QPY,
      QDMI_PROGRAM_FORMAT_IQMJSON,
      QDMI_PROGRAM_FORMAT_BATCHJOB,
      QDMI_PROGRAM_FORMAT_CUSTOM1,
      QDMI_PROGRAM_FORMAT_CUSTOM2,
      QDMI_PROGRAM_FORMAT_CUSTOM3,
      QDMI_PROGRAM_FORMAT_CUSTOM4,
      QDMI_PROGRAM_FORMAT_CUSTOM5,
  };

  for (const auto format : formats) {
    EXPECT_EQ(qdmi::isBinaryProgramFormat(format), expected(format))
        << "program format " << static_cast<int>(format);
  }
}

TEST_P(DeviceTest, Name) {
  EXPECT_NO_THROW(EXPECT_FALSE(mqt::test::value(device.getName()).empty()));
}

TEST_P(DeviceTest, Version) {
  EXPECT_NO_THROW(EXPECT_FALSE(mqt::test::value(device.getVersion()).empty()));
}

TEST_P(DeviceTest, Status) {
  EXPECT_NO_THROW(std::ignore = mqt::test::value(device.getStatus()));
}

TEST_P(DeviceTest, LibraryVersion) {
  EXPECT_NO_THROW(
      EXPECT_FALSE(mqt::test::value(device.getLibraryVersion()).empty()));
}

TEST_P(DeviceTest, QubitsNum) {
  EXPECT_NO_THROW(EXPECT_GT(mqt::test::value(device.getQubitsNum()), 0));
}

TEST_P(DeviceTest, Sites) {
  EXPECT_NO_THROW(EXPECT_FALSE(mqt::test::value(device.getSites()).empty()));
}

TEST_P(DeviceTest, CouplingMap) {
  EXPECT_NO_THROW(std::ignore = mqt::test::value(device.getCouplingMap()));
}

TEST_P(DeviceTest, NeedsCalibration) {
  EXPECT_NO_THROW(std::ignore = mqt::test::value(device.getNeedsCalibration()));
}

#ifdef MQT_CORE_QDMI_HAS_DDSIM_DEVICE
TEST_F(DDSimulatorDeviceTest, QueueLengthIsUnavailable) {
  EXPECT_EQ(mqt::test::value(device.getQueueLength()), std::nullopt);
}
#endif

TEST_P(DeviceTest, LengthUnit) {
  EXPECT_NO_THROW(std::ignore = mqt::test::value(device.getLengthUnit()));
}

TEST_P(DeviceTest, LengthScaleFactor) {
  EXPECT_NO_THROW(std::ignore =
                      mqt::test::value(device.getLengthScaleFactor()));
}

TEST_P(DeviceTest, DurationUnit) {
  EXPECT_NO_THROW(std::ignore = mqt::test::value(device.getDurationUnit()));
}

TEST_P(DeviceTest, DurationScaleFactor) {
  EXPECT_NO_THROW(std::ignore =
                      mqt::test::value(device.getDurationScaleFactor()));
}

TEST_P(DeviceTest, MinAtomDistance) {
  EXPECT_NO_THROW(std::ignore = mqt::test::value(device.getMinAtomDistance()));
}

TEST_P(DeviceTest, SupportedProgramFormats) {
  EXPECT_NO_THROW(std::ignore =
                      mqt::test::value(device.getSupportedProgramFormats()));
}

TEST_P(DeviceTest, ChildDevices) {
  EXPECT_TRUE(mqt::test::value(device.getChildDevices()).empty());
}

TEST_P(DeviceTest, UnsupportedCustomPropertyReturnsNullopt) {
  EXPECT_EQ(mqt::test::value(device.queryCustomProperty<std::vector<std::byte>>(
                CustomProperty::Custom2)),
            std::nullopt);
}

#ifdef MQT_CORE_QDMI_HAS_DDSIM_DEVICE
TEST_F(DDSimulatorDeviceTest, ReportsCompilerTargetMetadata) {
  EXPECT_EQ(mqt::test::value(device.queryCustomProperty<std::string>(
                CustomProperty::Custom1)),
            "mqt.compiler-target.v1:all-to-all-homogeneous");
}
#endif

TEST_P(SiteTest, Index) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getIndex()));
  }
}

TEST_P(SiteTest, T1) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getT1()));
  }
}

TEST_P(SiteTest, T2) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getT2()));
  }
}

TEST_P(SiteTest, Name) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getName()));
  }
}

TEST_P(SiteTest, XCoordinate) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getXCoordinate()));
  }
}

TEST_P(SiteTest, YCoordinate) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getYCoordinate()));
  }
}

TEST_P(SiteTest, ZCoordinate) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getZCoordinate()));
  }
}

TEST_P(SiteTest, IsZone) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.isZone()));
  }
}

TEST_P(SiteTest, XExtent) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getXExtent()));
  }
}

TEST_P(SiteTest, YExtent) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getYExtent()));
  }
}

TEST_P(SiteTest, ZExtent) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getZExtent()));
  }
}

TEST_P(SiteTest, ModuleIndex) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getModuleIndex()));
  }
}

TEST_P(SiteTest, SubmoduleIndex) {
  for (const auto& site : sites) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(site.getSubmoduleIndex()));
  }
}

TEST_P(SiteTest, UnsupportedCustomPropertyReturnsNullopt) {
  for (const auto& site : sites) {
    EXPECT_EQ(mqt::test::value(site.queryCustomProperty<std::vector<std::byte>>(
                  CustomProperty::Custom1)),
              std::nullopt);
  }
}

TEST_P(OperationTest, Name) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(
        EXPECT_FALSE(mqt::test::value(operation.getName()).empty()));
  }
}

TEST_P(OperationTest, QubitsNum) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(operation.getQubitsNum()));
  }
}

TEST_P(OperationTest, ParametersNum) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore =
                        mqt::test::value(operation.getParametersNum()));
  }
}

TEST_P(OperationTest, Duration) {
  for (const auto& operation : operations) {
    const auto qubitsNum = mqt::test::value(operation.getQubitsNum());
    if (!qubitsNum.has_value()) {
      EXPECT_NO_THROW(std::ignore = mqt::test::value(operation.getDuration()));
      continue;
    }
    const auto numQubits = *qubitsNum;
    if (numQubits == 1) {
      const auto sites = mqt::test::value(operation.getSites());
      if (!sites.has_value()) {
        EXPECT_NO_THROW(std::ignore =
                            mqt::test::value(operation.getDuration()));
        continue;
      }
      for (const auto& site : *sites) {
        EXPECT_NO_THROW(std::ignore =
                            mqt::test::value(operation.getDuration({site})));
      }
      continue;
    }

    if (numQubits == 2) {
      const auto sitePairs = mqt::test::value(operation.getSitePairs());
      if (!sitePairs.has_value()) {
        EXPECT_NO_THROW(std::ignore =
                            mqt::test::value(operation.getDuration()));
        continue;
      }
      for (const auto& [site1, site2] : *sitePairs) {
        EXPECT_NO_THROW(std::ignore = mqt::test::value(
                            operation.getDuration({site1, site2})));
      }
      continue;
    }

    EXPECT_NO_THROW(std::ignore = mqt::test::value(operation.getDuration()));
  }
}

TEST_P(OperationTest, Fidelity) {
  for (const auto& operation : operations) {
    const auto qubitsNum = mqt::test::value(operation.getQubitsNum());
    if (!qubitsNum.has_value()) {
      EXPECT_NO_THROW(std::ignore = mqt::test::value(operation.getFidelity()));
      continue;
    }
    const auto numQubits = *qubitsNum;
    if (numQubits == 1) {
      const auto sites = mqt::test::value(operation.getSites());
      if (!sites.has_value()) {
        EXPECT_NO_THROW(std::ignore =
                            mqt::test::value(operation.getFidelity()));
        continue;
      }
      for (const auto& site : *sites) {
        EXPECT_NO_THROW(std::ignore =
                            mqt::test::value(operation.getFidelity({site})));
      }
      continue;
    }

    if (numQubits == 2) {
      const auto sitePairs = mqt::test::value(operation.getSitePairs());
      if (!sitePairs.has_value()) {
        EXPECT_NO_THROW(std::ignore =
                            mqt::test::value(operation.getFidelity()));
        continue;
      }
      for (const auto& [site1, site2] : *sitePairs) {
        EXPECT_NO_THROW(std::ignore = mqt::test::value(
                            operation.getFidelity({site1, site2})));
      }
      continue;
    }

    EXPECT_NO_THROW(std::ignore = mqt::test::value(operation.getFidelity()));
  }
}

TEST_P(OperationTest, InteractionRadius) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore =
                        mqt::test::value(operation.getInteractionRadius()));
  }
}

TEST_P(OperationTest, BlockingRadius) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore =
                        mqt::test::value(operation.getBlockingRadius()));
  }
}

TEST_P(OperationTest, IdlingFidelity) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore =
                        mqt::test::value(operation.getIdlingFidelity()));
  }
}

TEST_P(OperationTest, IsZoned) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(operation.isZoned()));
  }
}

TEST_P(OperationTest, Sites) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore = mqt::test::value(operation.getSites()));
  }
}

TEST_P(OperationTest, SitePairs) {
  for (const auto& operation : operations) {
    const auto sitePairs = mqt::test::value(operation.getSitePairs());
    const auto qubitsNum = mqt::test::value(operation.getQubitsNum());
    const auto isZonedOp = mqt::test::value(operation.isZoned());

    if (!qubitsNum.has_value() || *qubitsNum != 2 || isZonedOp) {
      EXPECT_FALSE(sitePairs.has_value());
      continue;
    }

    const auto sites = mqt::test::value(operation.getSites());
    if (!sites.has_value() || sites->empty() || sites->size() % 2 != 0) {
      EXPECT_FALSE(sitePairs.has_value());
      continue;
    }

    EXPECT_TRUE(sitePairs.has_value());
    if (sitePairs.has_value()) {
      EXPECT_EQ(sitePairs->size(), sites->size() / 2);
    }
  }
}

TEST_P(OperationTest, MeanShuttlingSpeed) {
  for (const auto& operation : operations) {
    EXPECT_NO_THROW(std::ignore =
                        mqt::test::value(operation.getMeanShuttlingSpeed()));
  }
}

TEST_P(OperationTest, UnsupportedCustomPropertyReturnsNullopt) {
  for (const auto& operation : operations) {
    EXPECT_EQ(
        mqt::test::value(operation.queryCustomProperty<std::vector<std::byte>>(
            CustomProperty::Custom2)),
        std::nullopt);
  }
}

TEST_P(DeviceTest, RegularSitesAndZones) {
  const auto allSites = mqt::test::value(device.getSites());
  const auto regularSites = mqt::test::value(device.getRegularSites());
  const auto zones = mqt::test::value(device.getZones());

  EXPECT_FALSE(allSites.empty());
  EXPECT_EQ(regularSites.size() + zones.size(), allSites.size());

  for (const auto& site : regularSites) {
    EXPECT_FALSE(mqt::test::value(site.isZone()));
  }

  for (const auto& site : zones) {
    EXPECT_TRUE(mqt::test::value(site.isZone()));
  }
}

#ifdef MQT_CORE_QDMI_HAS_DDSIM_DEVICE
TEST_F(DDSimulatorDeviceTest, SubmitJobReturnsValidJob) {
  const std::string qasm3Program = R"(
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;)";

  const auto job = mqt::test::value(
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 100));

  EXPECT_FALSE(mqt::test::value(job.getId()).empty());
  EXPECT_EQ(mqt::test::value(job.getProgramFormat()),
            QDMI_PROGRAM_FORMAT_QASM3);
  EXPECT_STREQ(mqt::test::value(job.getProgram()).c_str(),
               qasm3Program.c_str());
  EXPECT_EQ(mqt::test::value(job.getNumShots()), 100);
  EXPECT_TRUE(mqt::test::value(job.wait()));
  EXPECT_EQ(mqt::test::value(job.check()), QDMI_JOB_STATUS_DONE);
}

TEST_F(DDSimulatorDeviceTest, SubmitJobRejectsIncompatiblePayloadKinds) {
  const std::string textProgram = "OPENQASM 3.0;";

  EXPECT_EQ(mqt::test::errorStatus(device.submitJob(
                textProgram, QDMI_PROGRAM_FORMAT_QIRBASEMODULE, 0)),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSimulatorDeviceTest, SubmitJobRejectsBatchJobs) {
  // A batch job's program is a list of job handles, which the byte-span API
  // cannot express, so MQT Core states that it does not support them.
  constexpr std::array bytes{std::byte{0}};

  EXPECT_EQ(mqt::test::errorStatus(
                device.submitJob(bytes, QDMI_PROGRAM_FORMAT_BATCHJOB, 0)),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(mqt::test::errorStatus(device.submitJob(
                std::string{}, QDMI_PROGRAM_FORMAT_BATCHJOB, 0)),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSimulatorDeviceTest, SubmitJobSendsCalibrationRunsElsewhere) {
  // A calibration run takes no shot count and an optional payload, so it has
  // its own entry point rather than a special case in `submitJob`.
  EXPECT_EQ(mqt::test::errorStatus(device.submitJob(
                std::string{}, QDMI_PROGRAM_FORMAT_CALIBRATION, 0)),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSimulatorDeviceTest, CalibrationJobReachesTheDevice) {
  /// DDSIM rejects calibration with a device error. The client must forward
  /// the request rather than reject its optional payload as an invalid
  /// argument.
  EXPECT_TRUE(
      mqt::test::errorStatus(device.submitCalibrationJob()).has_value());
  EXPECT_TRUE(
      mqt::test::errorStatus(device.submitCalibrationJob("configuration"))
          .has_value());

  constexpr std::array payload{std::byte{1}, std::byte{2}};
  EXPECT_TRUE(
      mqt::test::errorStatus(device.submitCalibrationJob(payload)).has_value());

  constexpr std::byte emptyPayloadStorage{};
  const std::span emptyPayload{&emptyPayloadStorage, size_t{0}};
  EXPECT_TRUE(mqt::test::errorStatus(device.submitCalibrationJob(emptyPayload))
                  .has_value());

  EXPECT_NE(mqt::test::errorStatus(device.submitCalibrationJob()),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(DDSimulatorDeviceTest, SubmitJobCustomSupportedTypes) {
  const auto submitWithCustoms = [&](const CustomJobParameter& custom,
                                     size_t which) {
    std::array<std::optional<CustomJobParameter>, 5> parameters;
    parameters[which - 1] = custom;
    return device.submitJob("OPENQASM 3.0;", QDMI_PROGRAM_FORMAT_QASM3, 10,
                            parameters[0], parameters[1], parameters[2],
                            parameters[3], parameters[4]);
  };
  EXPECT_NO_THROW(mqt::test::value(submitWithCustoms(7, 1)));
  EXPECT_NO_THROW(mqt::test::value(submitWithCustoms(false, 2)));
  EXPECT_TRUE(mqt::test::errorStatus(submitWithCustoms(true, 2)));
  for (const CustomJobParameter& value : {
           CustomJobParameter(std::string("custom")),
           CustomJobParameter(42),
           CustomJobParameter(3.14),
       }) {
    EXPECT_EQ(mqt::test::errorStatus(submitWithCustoms(value, 2)),
              QDMI_ERROR_INVALIDARGUMENT);
  }
  for (size_t i = 3; i <= 5; ++i) {
    for (const CustomJobParameter& value : {
             CustomJobParameter(std::string("custom")),
             CustomJobParameter(42),
             CustomJobParameter(3.14),
             CustomJobParameter(true),
         }) {
      EXPECT_EQ(mqt::test::errorStatus(submitWithCustoms(value, i)),
                QDMI_ERROR_NOTSUPPORTED);
    }
  }
}

TEST_F(DDSimulatorDeviceTest, SubmitJobPreservesNumShots) {
  const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
)";

  const auto job1 = mqt::test::value(
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 10));
  EXPECT_EQ(mqt::test::value(job1.getNumShots()), 10);

  const auto job2 = mqt::test::value(
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 100));
  EXPECT_EQ(mqt::test::value(job2.getNumShots()), 100);

  const auto job3 = mqt::test::value(
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 1000));
  EXPECT_EQ(mqt::test::value(job3.getNumShots()), 1000);
}

TEST_F(JobTest, IdIsUnique) {
  const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
)";
  const auto job2 = mqt::test::value(
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 10));

  EXPECT_NE(mqt::test::value(job.getId()), mqt::test::value(job2.getId()));
}

TEST_F(JobTest, QueuePositionIsUnavailable) {
  EXPECT_EQ(mqt::test::value(job.getQueuePosition()), std::nullopt);
}

TEST_F(JobTest, UnsupportedCustomPropertyAndResultReturnNullopt) {
  EXPECT_EQ(mqt::test::value(job.queryCustomProperty<std::vector<std::byte>>(
                CustomProperty::Custom1)),
            std::nullopt);
  EXPECT_TRUE(mqt::test::value(job.wait()));
  EXPECT_EQ(mqt::test::value(job.getCustomResult<std::vector<std::byte>>(
                CustomProperty::Custom1)),
            std::nullopt);
}

TEST_F(JobTest, StatusProgresses) {
  EXPECT_TRUE(mqt::test::value(job.wait()));

  const auto finalStatus = mqt::test::value(job.check());
  EXPECT_THAT(finalStatus,
              testing::AnyOf(QDMI_JOB_STATUS_DONE, QDMI_JOB_STATUS_FAILED));
}

TEST_F(JobTest, GetCountsReturnsValidHistogram) {
  EXPECT_TRUE(mqt::test::value(job.wait()));

  const auto counts = mqt::test::value(job.getCounts());
  EXPECT_FALSE(counts.empty());

  // All keys should be valid binary strings of length 1 (single qubit)
  for (const auto& [key, value] : counts) {
    EXPECT_EQ(key.length(), 1);
    EXPECT_TRUE(key == "0" || key == "1");
    EXPECT_GT(value, 0);
  }

  size_t totalCounts = 0;
  for (const auto& value : counts | std::views::values) {
    totalCounts += value;
  }
  EXPECT_EQ(totalCounts, mqt::test::value(job.getNumShots()));
}

TEST_F(JobTest, MultipleGetCountsCalls) {
  EXPECT_TRUE(mqt::test::value(job.wait()));

  const auto counts1 = mqt::test::value(job.getCounts());
  const auto counts2 = mqt::test::value(job.getCounts());

  EXPECT_EQ(counts1, counts2);
}

TEST_F(JobTest, GetShotsReturnsValidShots) {
  EXPECT_TRUE(mqt::test::value(job.wait()));
  auto result = job.getShots();
  if (const auto* error = std::get_if<Error>(&result)) {
    EXPECT_EQ(error->status, QDMI_ERROR_NOTSUPPORTED);
    return;
  }
  const auto& shots = std::get<0>(result);
  EXPECT_FALSE(shots.empty());
  for (const auto& shot : shots) {
    EXPECT_EQ(shot.length(), 1);
    EXPECT_TRUE(shot == "0" || shot == "1");
  }
  EXPECT_EQ(shots.size(), mqt::test::value(job.getNumShots()));
}

TEST_F(JobTest, CancelJob) {
  const std::string qasm3Program = R"(
OPENQASM 3.0;
qubit[1] q;
bit[1] c;
c[0] = measure q[0];
)";
  const auto jobToCancel = mqt::test::value(
      device.submitJob(qasm3Program, QDMI_PROGRAM_FORMAT_QASM3, 10));

  const auto error = jobToCancel.cancel();
  const auto status = mqt::test::value(jobToCancel.check());
  if (error) {
    EXPECT_EQ(error->status, QDMI_ERROR_INVALIDARGUMENT);
    EXPECT_THAT(status,
                testing::AnyOf(QDMI_JOB_STATUS_DONE, QDMI_JOB_STATUS_FAILED));
  } else {
    EXPECT_EQ(status, QDMI_JOB_STATUS_CANCELED);
  }
}

TEST_F(JobTest, CancelCompletedJobReturnsError) {
  EXPECT_TRUE(mqt::test::value(job.wait()));

  const auto statusBefore = mqt::test::value(job.check());
  EXPECT_THAT(statusBefore,
              testing::AnyOf(QDMI_JOB_STATUS_DONE, QDMI_JOB_STATUS_FAILED));

  EXPECT_EQ(mqt::test::errorStatus(job.cancel()), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_F(SimulatorJobTest, getDenseStateVectorReturnsValidState) {
  EXPECT_TRUE(mqt::test::value(job.wait()));

  const auto stateVector = mqt::test::value(job.getDenseStateVector());
  EXPECT_EQ(stateVector.size(), 4); // 2 qubits → 4 amplitudes

  // The expected state is (|00⟩ + |11⟩)/sqrt(2)
  constexpr double invSqrt2 = 1.0 / std::numbers::sqrt2;
  EXPECT_NEAR(std::abs(stateVector[0]), invSqrt2, 1e-10); // |00⟩
  EXPECT_NEAR(std::abs(stateVector[1]), 0.0, 1e-10);      // |01⟩
  EXPECT_NEAR(std::abs(stateVector[2]), 0.0, 1e-10);
  EXPECT_NEAR(std::abs(stateVector[3]), invSqrt2, 1e-10); // |11⟩
}

TEST_F(SimulatorJobTest, getDenseProbabilitiesReturnsValidProbabilities) {
  EXPECT_TRUE(mqt::test::value(job.wait()));

  const auto probabilities = mqt::test::value(job.getDenseProbabilities());
  EXPECT_EQ(probabilities.size(), 4); // 2 qubits → 4 probabilities

  // The expected probabilities are 0.5 for |00⟩ and |11⟩, and 0 for |01⟩ and
  // |10⟩
  EXPECT_NEAR(probabilities[0], 0.5, 1e-10); // |00⟩
  EXPECT_NEAR(probabilities[1], 0.0, 1e-10); // |01⟩
  EXPECT_NEAR(probabilities[2], 0.0, 1e-10); // |10⟩
  EXPECT_NEAR(probabilities[3], 0.5, 1e-10); // |11⟩
}

TEST_F(SimulatorJobTest, getSparseStateVectorReturnsValidState) {
  EXPECT_TRUE(mqt::test::value(job.wait()));

  const auto sparseStateVector = mqt::test::value(job.getSparseStateVector());
  EXPECT_EQ(sparseStateVector.size(),
            2); // Only |00⟩ and |11⟩ should be present

  constexpr double invSqrt2 = 1.0 / std::numbers::sqrt2;
  const auto it00 = sparseStateVector.find("00");
  ASSERT_NE(it00, sparseStateVector.end());
  EXPECT_NEAR(std::abs(it00->second), invSqrt2, 1e-10);

  const auto it11 = sparseStateVector.find("11");
  ASSERT_NE(it11, sparseStateVector.end());
  EXPECT_NEAR(std::abs(it11->second), invSqrt2, 1e-10);
}

TEST_F(SimulatorJobTest, getSparseProbabilitiesReturnsValidProbabilities) {
  EXPECT_TRUE(mqt::test::value(job.wait()));

  const auto sparseProbabilities =
      mqt::test::value(job.getSparseProbabilities());
  EXPECT_EQ(sparseProbabilities.size(),
            2); // Only |00⟩ and |11⟩ should be present

  const auto it00 = sparseProbabilities.find("00");
  ASSERT_NE(it00, sparseProbabilities.end());
  EXPECT_NEAR(it00->second, 0.5, 1e-10);

  const auto it11 = sparseProbabilities.find("11");
  ASSERT_NE(it11, sparseProbabilities.end());
  EXPECT_NEAR(it11->second, 0.5, 1e-10);
}
#endif

TEST(AuthenticationTest, SessionParameterToString) {
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_TOKEN), "TOKEN");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_AUTHFILE), "AUTH FILE");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_AUTHURL), "AUTH URL");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_USERNAME), "USERNAME");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_PASSWORD), "PASSWORD");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_PROJECTID), "PROJECT ID");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_MAX), "MAX");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_CUSTOM1), "CUSTOM1");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_CUSTOM2), "CUSTOM2");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_CUSTOM3), "CUSTOM3");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_CUSTOM4), "CUSTOM4");
  EXPECT_STREQ(qdmi::toString(QDMI_SESSION_PARAMETER_CUSTOM5), "CUSTOM5");
}

TEST(AuthenticationTest, SessionConstructionWithToken) {
  // Empty token should be accepted
  SessionConfig config1;
  config1.token = "";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config1));
  });

  // Non-empty token should be accepted
  SessionConfig config2;
  config2.token = "test_token_123";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config2));
  });

  // Token with special characters should be accepted
  SessionConfig config3;
  config3.token = "very_long_token_with_special_characters_!@#$%^&*()";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config3));
  });
}

TEST(AuthenticationTest, ReportsSkippedUnsupportedParameter) {
  SessionConfig config;
  config.token = "test-token";

  testing::internal::CaptureStderr();
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config));
  });
  const auto diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_THAT(
      diagnostic,
      testing::AllOf(testing::HasSubstr("[mqt-core] [info]"),
                     testing::HasSubstr(
                         "Session parameter TOKEN not supported (skipped)")));
}

TEST(AuthenticationTest, SessionConstructionWithAuthUrl) {
  // Valid HTTPS URL
  SessionConfig config1;
  config1.authUrl = "https://example.com";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config1));
  });

  // Valid HTTP URL with port and path
  SessionConfig config2;
  config2.authUrl = "http://auth.server.com:8080/api";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config2));
  });

  // Valid HTTPS URL with query parameters
  SessionConfig config3;
  config3.authUrl = "https://auth.example.com/token?param=value";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config3));
  });

  // Valid localhost URL
  SessionConfig configLocalhost;
  configLocalhost.authUrl = "http://localhost";
  EXPECT_NO_THROW({
    const auto session =
        mqt::test::value(qdmi::Session::create(configLocalhost));
  });

  // Valid localhost URL with port
  SessionConfig configLocalhostPort;
  configLocalhostPort.authUrl = "http://localhost:8080";
  EXPECT_NO_THROW({
    const auto session =
        mqt::test::value(qdmi::Session::create(configLocalhostPort));
  });

  // Valid localhost URL with port and path
  SessionConfig configLocalhostPath;
  configLocalhostPath.authUrl = "https://localhost:3000/auth/api";
  EXPECT_NO_THROW({
    const auto session =
        mqt::test::value(qdmi::Session::create(configLocalhostPath));
  });

  // Valid IPv4 address URL
  SessionConfig configIPv4;
  configIPv4.authUrl = "http://127.0.0.1:5000/auth";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(configIPv4));
  });

  // Valid IPv6 address URL
  SessionConfig configIPv6;
  configIPv6.authUrl = "https://[::1]:8080/auth";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(configIPv6));
  });

  // Invalid URL - not a URL at all (validation fails before setting parameter)
  SessionConfig config4;
  config4.authUrl = "not-a-url";
  EXPECT_TRUE(
      mqt::test::errorStatus(qdmi::Session::create(config4)).has_value());

  // Invalid URL - unsupported protocol
  SessionConfig config5;
  config5.authUrl = "ftp://invalid.com";
  EXPECT_TRUE(
      mqt::test::errorStatus(qdmi::Session::create(config5)).has_value());

  // Invalid URL - missing protocol
  SessionConfig config6;
  config6.authUrl = "example.com";
  EXPECT_TRUE(
      mqt::test::errorStatus(qdmi::Session::create(config6)).has_value());

  // Invalid URL - empty
  SessionConfig config7;
  config7.authUrl = "";
  EXPECT_TRUE(
      mqt::test::errorStatus(qdmi::Session::create(config7)).has_value());
}

TEST(AuthenticationTest, SessionConstructionWithAuthFile) {
  // Non-existent file (validation fails before setting parameter)
  SessionConfig config1;
  config1.authFile = "/nonexistent/path/to/file.txt";
  EXPECT_TRUE(
      mqt::test::errorStatus(qdmi::Session::create(config1)).has_value());

  // Existing file (should succeed even if parameter is unsupported)
  const auto tempDir = std::filesystem::temp_directory_path();
  auto const tmpPath = tempDir / ("qdmi_test_auth_" +
                                  std::to_string(std::hash<std::thread::id>{}(
                                      std::this_thread::get_id())) +
                                  ".txt");
  {
    std::ofstream tmpFile(tmpPath);
    ASSERT_TRUE(tmpFile.is_open()) << "Failed to create temporary file";
    tmpFile << "test_token_content";
  }

  SessionConfig config2;
  config2.authFile = tmpPath;
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config2));
  });

  // Clean up
  std::filesystem::remove(tmpPath);
}

TEST(AuthenticationTest, SessionConstructionWithUsernamePassword) {
  // Username only
  SessionConfig config1;
  config1.username = "user123";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config1));
  });

  // Password only
  SessionConfig config2;
  config2.password = "secure_password";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config2));
  });

  // Both username and password
  SessionConfig config3;
  config3.username = "user123";
  config3.password = "secure_password";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config3));
  });
}

TEST(AuthenticationTest, SessionConstructionWithProjectId) {
  SessionConfig config;
  config.projectId = "project-123-abc";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config));
  });
}

TEST(AuthenticationTest, SessionConstructionWithMultipleParameters) {
  SessionConfig config;
  config.token = "test_token";
  config.username = "test_user";
  config.password = "test_pass";
  config.projectId = "test_project";
  EXPECT_NO_THROW({
    const auto session = mqt::test::value(qdmi::Session::create(config));
  });
}

TEST(AuthenticationTest, SessionConstructionWithCustomParameters) {
  // Custom parameters may not be supported by all devices, or may have specific
  // validation requirements. This test verifies they can be passed to the
  // Session constructor. Currently a smoke test.

  // Test custom1 - may succeed or fail with validation/unsupported errors
  SessionConfig config1;
  config1.custom1 = "custom_value_1";
  auto session1 = mqt::test::value(qdmi::Session::create(config1));
  EXPECT_NO_THROW(std::ignore = mqt::test::value(session1.getDevices()));

  // Test custom2
  SessionConfig config2;
  config2.custom2 = "custom_value_2";
  auto session2 = mqt::test::value(qdmi::Session::create(config2));
  EXPECT_NO_THROW(std::ignore = mqt::test::value(session2.getDevices()));

  // Test all custom parameters together
  SessionConfig config3;
  config3.custom1 = "value1";
  config3.custom2 = "value2";
  config3.custom3 = "value3";
  config3.custom4 = "value4";
  config3.custom5 = "value5";
  auto session3 = mqt::test::value(qdmi::Session::create(config3));
  EXPECT_NO_THROW(std::ignore = mqt::test::value(session3.getDevices()));

  // Test mixing custom parameters with standard authentication
  SessionConfig config4;
  config4.token = "test_token";
  config4.custom1 = "custom_value";
  config4.projectId = "project_id";
  auto session4 = mqt::test::value(qdmi::Session::create(config4));
  EXPECT_NO_THROW(std::ignore = mqt::test::value(session4.getDevices()));
}

TEST(AuthenticationTest, SessionGetDevicesReturnsList) {
  auto session = mqt::test::value(qdmi::Session::create());
  auto const devices = mqt::test::value(session.getDevices());

  EXPECT_FALSE(devices.empty());

  // All elements should be Device instances
  for (const auto& device : devices) {
    // Device should have a name
    EXPECT_FALSE(mqt::test::value(device.getName()).empty());
  }
}

TEST(AuthenticationTest, SessionMultipleInstances) {
  auto session1 = mqt::test::value(qdmi::Session::create());
  auto session2 = mqt::test::value(qdmi::Session::create());

  auto const devices1 = mqt::test::value(session1.getDevices());
  auto const devices2 = mqt::test::value(session2.getDevices());

  // Both should return devices
  EXPECT_FALSE(devices1.empty());
  EXPECT_FALSE(devices2.empty());

  // Should return the same number of devices
  EXPECT_EQ(devices1.size(), devices2.size());
}

TEST(DeviceOwnershipTest, SiteKeepsFreshSessionAlive) {
  const auto site = [] {
    auto const device = mqt::test::value(Session::openDevice("mqt.sc.default"));
    return mqt::test::value(device.getSites()).front();
  }();

  EXPECT_EQ(mqt::test::value(site.getIndex()), 0);
}

TEST(DeviceOwnershipTest, OperationKeepsFreshSessionAlive) {
  const auto operation = [] {
    auto const device = mqt::test::value(Session::openDevice("mqt.sc.default"));
    return mqt::test::value(device.getOperations()).front();
  }();

  EXPECT_FALSE(mqt::test::value(operation.getName()).empty());
}

TEST(DeviceOwnershipTest, SiteFromOperationKeepsFreshSessionAlive) {
  const auto site = [] {
    auto const device = mqt::test::value(Session::openDevice("mqt.sc.default"));
    const auto operation = mqt::test::value(device.getOperations()).front();
    return mqt::test::value(operation.getSites()).value().front();
  }();

  EXPECT_FALSE(mqt::test::value(site.isZone()));
}

namespace {
// Helper function to get all devices for parameterized tests
auto getDevices() -> std::vector<Device> {
  auto session = mqt::test::value(qdmi::Session::create());
  return mqt::test::value(session.getDevices());
}
} // namespace

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    DeviceTest,
    // Test suite name
    DeviceTest,
    // Parameters to test with
    testing::ValuesIn(getDevices()),
    [](const testing::TestParamInfo<Device>& paramInfo) {
      auto name = mqt::test::value(paramInfo.param.getName());
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    SiteTest,
    // Test suite name
    SiteTest,
    // Parameters to test with
    testing::ValuesIn(getDevices()),
    [](const testing::TestParamInfo<Device>& paramInfo) {
      auto name = mqt::test::value(paramInfo.param.getName());
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    OperationTest,
    // Test suite name
    OperationTest,
    // Parameters to test with
    testing::ValuesIn(getDevices()),
    [](const testing::TestParamInfo<Device>& paramInfo) {
      auto name = mqt::test::value(paramInfo.param.getName());
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      return name;
    });

} // namespace qdmi
