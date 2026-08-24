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
#include "qdmi/driver/Driver.hpp"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <qdmi/client.h>
#include <qdmi/device.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

/// The private Driver ABI fixes these exported symbol names.
/// NOLINTBEGIN(readability-identifier-naming)
extern "C" int MQT_CORE_QDMI_driver_add_manifest_v1(const char* manifestPath);
extern "C" int MQT_CORE_QDMI_driver_session_alloc_for_device_v1(
    const char* deviceId, size_t deviceSessionJsonSize,
    const char* deviceSessionJson, QDMI_Session* session);
/// NOLINTEND(readability-identifier-naming)

namespace testing {
namespace {
auto stringConcat5(const std::string& a, const std::string& b,
                   const std::string& c, const std::string& d,
                   const std::string& e) -> std::string {
  std::stringstream ss;
  ss << a << b << c << d << e;
  return ss.str();
}
// NOLINTBEGIN(readability-identifier-naming,cppcoreguidelines-avoid-const-or-ref-data-members)
MATCHER_P2(IsBetween, a, b,
           stringConcat5(negation ? "isn't" : "is", " between ",
                         PrintToString(a), " and ", PrintToString(b))) {
  return a <= arg && arg <= b;
}
// NOLINTEND(readability-identifier-naming,cppcoreguidelines-avoid-const-or-ref-data-members)
} // namespace
} // namespace testing

namespace {

struct ConfiguredDriverEnvironment {
  ConfiguredDriverEnvironment() noexcept {
#ifdef _WIN32
    if (_putenv_s("MQT_CORE_QDMI_CONFIG_FILE",
                  MQT_CORE_QDMI_TEST_CONFIG_FILE) != 0) {
#else
    // POSIX exposes setenv through <cstdlib>, but include-cleaner does not
    // associate the global declaration with that C++ header.
    // NOLINTNEXTLINE(misc-include-cleaner)
    if (setenv("MQT_CORE_QDMI_CONFIG_FILE", MQT_CORE_QDMI_TEST_CONFIG_FILE,
               1) != 0) {
#endif
      std::abort();
    }
  }
};

const ConfiguredDriverEnvironment CONFIGURED_DRIVER_ENVIRONMENT;

class ChildDeviceLibrary final : public qdmi::DeviceLibrary {
  struct Child {
    size_t id;
  };

  struct Session {
    ChildDeviceLibrary* library = nullptr;
    QDMI_Child_Device child = nullptr;
    bool initialized = false;
  };

  static inline ChildDeviceLibrary* activeLibrary = nullptr;
  std::array<Child, 2> children_{{{0}, {1}}};
  std::unordered_map<QDMI_Device_Session, std::unique_ptr<Session>> sessions_;

  [[nodiscard]] static auto asSession(QDMI_Device_Session_impl_d* const session)
      -> Session* {
    return reinterpret_cast<Session*>(session);
  }

  static auto alloc(QDMI_Device_Session* session) -> int {
    if (session == nullptr || activeLibrary == nullptr) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto fakeSession =
        std::make_unique<Session>(Session{.library = activeLibrary});
    auto* const sessionPtr = fakeSession.get();
    auto* const sessionHandle =
        reinterpret_cast<QDMI_Device_Session>(sessionPtr);
    activeLibrary->sessions_.emplace(sessionHandle, std::move(fakeSession));
    ++activeLibrary->allocatedSessions;
    *session = sessionHandle;
    return QDMI_SUCCESS;
  }

  static void free(QDMI_Device_Session session) {
    if (session == nullptr) {
      return;
    }
    const auto* const fakeSession = asSession(session);
    ++fakeSession->library->freedSessions;
    fakeSession->library->sessions_.erase(session);
  }

  static auto setParameter(QDMI_Device_Session session,
                           const QDMI_Device_Session_Parameter parameter,
                           const size_t size, const void* value) -> int {
    if (session == nullptr || value == nullptr || size == 0) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    if (parameter != QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    auto* const fakeSession = asSession(session);
    if (fakeSession->library->rejectChildSelection ||
        size != sizeof(QDMI_Child_Device)) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    std::memcpy(static_cast<void*>(&fakeSession->child), value,
                sizeof(QDMI_Child_Device));
    fakeSession->library->selectedChildren.emplace_back(fakeSession->child);
    return QDMI_SUCCESS;
  }

  static auto init(QDMI_Device_Session session) -> int {
    if (session == nullptr) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    asSession(session)->initialized = true;
    return QDMI_SUCCESS;
  }

  static auto queryDeviceProperty(QDMI_Device_Session session,
                                  const QDMI_Device_Property property,
                                  const size_t size, void* value,
                                  size_t* sizeRet) -> int {
    if (session == nullptr || (value != nullptr && size == 0)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    const auto* const fakeSession = asSession(session);
    if (!fakeSession->initialized) {
      return QDMI_ERROR_BADSTATE;
    }

    if (property == QDMI_DEVICE_PROPERTY_CHILDDEVICES) {
      if (fakeSession->child != nullptr) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
      auto* const library = fakeSession->library;
      if (library->childDevicesNotSupported) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
      if (library->childDeviceQueryFails) {
        return QDMI_ERROR_BADSTATE;
      }
      const size_t requiredSize =
          library->malformedChildList
              ? sizeof(QDMI_Child_Device) + 1
              : library->children_.size() * sizeof(QDMI_Child_Device);
      if (sizeRet != nullptr) {
        *sizeRet = requiredSize;
      }
      if (value != nullptr) {
        if (size < requiredSize || library->malformedChildList) {
          return QDMI_ERROR_INVALIDARGUMENT;
        }
        std::array<QDMI_Child_Device, 2> handles{};
        std::ranges::transform(
            library->children_, handles.begin(), [](Child& child) {
              return reinterpret_cast<QDMI_Child_Device>(&child);
            });
        std::memcpy(value, static_cast<const void*>(handles.data()),
                    requiredSize);
      }
      return QDMI_SUCCESS;
    }

    if (property == QDMI_DEVICE_PROPERTY_NAME) {
      std::string name = "parent";
      if (fakeSession->child != nullptr) {
        const auto* child = reinterpret_cast<const Child*>(fakeSession->child);
        name = "child-" + std::to_string(child->id);
      }
      const auto requiredSize = name.size() + 1;
      if (sizeRet != nullptr) {
        *sizeRet = requiredSize;
      }
      if (value != nullptr) {
        if (size < requiredSize) {
          return QDMI_ERROR_INVALIDARGUMENT;
        }
        std::memcpy(value, name.c_str(), requiredSize);
      }
      return QDMI_SUCCESS;
    }
    return QDMI_ERROR_NOTSUPPORTED;
  }

public:
  size_t allocatedSessions = 0;
  size_t freedSessions = 0;
  bool rejectChildSelection = false;
  bool malformedChildList = false;
  bool childDevicesNotSupported = false;
  bool childDeviceQueryFails = false;
  std::vector<QDMI_Child_Device> selectedChildren;

  ChildDeviceLibrary() {
    activeLibrary = this;
    device_session_alloc = alloc;
    device_session_free = free;
    device_session_set_parameter = setParameter;
    device_session_init = init;
    device_session_query_device_property = queryDeviceProperty;
  }

  ~ChildDeviceLibrary() override { activeLibrary = nullptr; }

  [[nodiscard]] auto childHandle(const size_t index) -> QDMI_Child_Device {
    return reinterpret_cast<QDMI_Child_Device>(&children_.at(index));
  }
};

[[nodiscard]] auto queryTextProperty(QDMI_Device_impl_d* const device,
                                     const QDMI_Device_Property property)
    -> std::string {
  size_t size = 0;
  EXPECT_EQ(
      QDMI_device_query_device_property(device, property, 0, nullptr, &size),
      QDMI_SUCCESS);
  std::string name(size - 1, '\0');
  EXPECT_EQ(QDMI_device_query_device_property(device, property, size,
                                              name.data(), nullptr),
            QDMI_SUCCESS);
  return name;
}

[[nodiscard]] auto queryName(QDMI_Device_impl_d* const device) -> std::string {
  return queryTextProperty(device, QDMI_DEVICE_PROPERTY_NAME);
}

[[nodiscard]] auto openTestDevice(const std::string& library,
                                  const std::string& prefix,
                                  const qdmi::DeviceSessionConfig& session = {})
    -> QDMI_Device {
  static size_t nextId = 0;
  auto& driver = qdmi::Driver::get();
  const auto id = "test.runtime." + std::to_string(nextId++);
  driver.registerDevice(
      {.id = id, .library = library, .prefix = prefix, .session = session});
  return driver.open(id);
}

class DriverTest : public testing::TestWithParam<const char*> {
protected:
  QDMI_Session session = nullptr;
  QDMI_Device device = nullptr;

  void SetUp() override {
    const auto& deviceName = GetParam();

    ASSERT_EQ(QDMI_session_alloc(&session), QDMI_SUCCESS)
        << "Failed to allocate session.";

    ASSERT_EQ(QDMI_session_init(session), QDMI_SUCCESS)
        << "Failed to initialize session.";

    size_t devicesSize = 0;
    ASSERT_EQ(QDMI_session_query_session_property(session,
                                                  QDMI_SESSION_PROPERTY_DEVICES,
                                                  0, nullptr, &devicesSize),
              QDMI_SUCCESS)
        << "Failed to retrieve number of devices.";
    std::vector<QDMI_Device> devices(devicesSize / sizeof(QDMI_Device));
    ASSERT_EQ(QDMI_session_query_session_property(
                  session, QDMI_SESSION_PROPERTY_DEVICES, devicesSize,
                  static_cast<void*>(devices.data()), nullptr),
              QDMI_SUCCESS)
        << "Failed to retrieve devices.";

    for (auto* const dev : devices) {
      size_t namesSize = 0;
      ASSERT_EQ(QDMI_device_query_device_property(
                    dev, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &namesSize),
                QDMI_SUCCESS)
          << "Failed to retrieve the length of the device's name.";
      std::string name(namesSize - 1, '\0');
      ASSERT_EQ(
          QDMI_device_query_device_property(dev, QDMI_DEVICE_PROPERTY_NAME,
                                            namesSize, name.data(), nullptr),
          QDMI_SUCCESS)
          << "Failed to retrieve the device's name.";

      ASSERT_FALSE(name.empty()) << "Device must provide a non-empty name.";

      if (name == deviceName) {
        device = dev;
        return;
      }
    }
    FAIL() << "Device with name '" << deviceName
           << "' not found in the session.";
  }

  void TearDown() override { QDMI_session_free(session); }
};

class DriverJobTest : public DriverTest {
protected:
  QDMI_Job job = nullptr;

  void SetUp() override {
    DriverTest::SetUp();
    ASSERT_EQ(QDMI_device_create_job(device, &job), QDMI_SUCCESS)
        << "Failed to create a device job.";
  }

  void TearDown() override {
    if (job != nullptr) {
      QDMI_job_free(job);
      job = nullptr;
    }
    DriverTest::TearDown();
  }
};

TEST(ChildDeviceTest, WrapsOpaqueHandlesInStableClientDevices) {
  const auto library = std::make_shared<ChildDeviceLibrary>();
  {
    QDMI_Device_impl_d parent(library);
    size_t size = 0;
    ASSERT_EQ(
        QDMI_device_query_device_property(
            &parent, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr, &size),
        QDMI_SUCCESS);
    ASSERT_EQ(size, 2 * sizeof(QDMI_Device));

    std::array<QDMI_Device, 2> children{};
    ASSERT_EQ(QDMI_device_query_device_property(
                  &parent, QDMI_DEVICE_PROPERTY_CHILDDEVICES, size,
                  static_cast<void*>(children.data()), nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(queryName(children[0]), "child-0");
    EXPECT_EQ(queryName(children[1]), "child-1");

    std::array<QDMI_Device, 2> repeatedQuery{};
    ASSERT_EQ(QDMI_device_query_device_property(
                  &parent, QDMI_DEVICE_PROPERTY_CHILDDEVICES, size,
                  static_cast<void*>(repeatedQuery.data()), nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(repeatedQuery, children);
    EXPECT_EQ(library->selectedChildren,
              (std::vector{library->childHandle(0), library->childHandle(1)}));
    EXPECT_EQ(library->allocatedSessions, 3);
    EXPECT_EQ(library->freedSessions, 0);

    EXPECT_EQ(QDMI_device_query_device_property(
                  &parent, QDMI_DEVICE_PROPERTY_CHILDDEVICES,
                  sizeof(QDMI_Device), static_cast<void*>(children.data()),
                  nullptr),
              QDMI_ERROR_INVALIDARGUMENT);
    EXPECT_EQ(QDMI_device_query_device_property(
                  children[0], QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr,
                  nullptr),
              QDMI_ERROR_NOTSUPPORTED);
  }
  EXPECT_EQ(library->freedSessions, 3);
}

TEST(ChildDeviceTest, CleansUpWhenSelectingAChildFails) {
  const auto library = std::make_shared<ChildDeviceLibrary>();
  library->rejectChildSelection = true;
  EXPECT_THROW(QDMI_Device_impl_d{library}, std::runtime_error);
  EXPECT_EQ(library->allocatedSessions, 2);
  EXPECT_EQ(library->freedSessions, 2);
}

TEST(ChildDeviceTest, RejectsMalformedChildLists) {
  const auto library = std::make_shared<ChildDeviceLibrary>();
  library->malformedChildList = true;
  EXPECT_THROW(QDMI_Device_impl_d{library}, std::invalid_argument);
  EXPECT_EQ(library->allocatedSessions, 1);
  EXPECT_EQ(library->freedSessions, 1);
}

TEST(ChildDeviceTest, SupportsDevicesWithoutChildDevices) {
  const auto library = std::make_shared<ChildDeviceLibrary>();
  library->childDevicesNotSupported = true;
  {
    QDMI_Device_impl_d parent(library);
    size_t size = 0;
    EXPECT_EQ(
        QDMI_device_query_device_property(
            &parent, QDMI_DEVICE_PROPERTY_CHILDDEVICES, 0, nullptr, &size),
        QDMI_ERROR_NOTSUPPORTED);
    EXPECT_EQ(library->allocatedSessions, 1);
  }
  EXPECT_EQ(library->freedSessions, 1);
}

TEST(ChildDeviceTest, CleansUpWhenQueryingChildDevicesFails) {
  const auto library = std::make_shared<ChildDeviceLibrary>();
  library->childDeviceQueryFails = true;
  EXPECT_THROW(QDMI_Device_impl_d{library}, std::runtime_error);
  EXPECT_EQ(library->allocatedSessions, 1);
  EXPECT_EQ(library->freedSessions, 1);
}

TEST_P(DriverTest, SessionSetParameter) {
  const std::string authFile = "authfile.txt";
  QDMI_Session uninitializedSession = nullptr;
  ASSERT_EQ(QDMI_session_alloc(&uninitializedSession), QDMI_SUCCESS);
  EXPECT_EQ(QDMI_session_set_parameter(uninitializedSession,
                                       QDMI_SESSION_PARAMETER_AUTHFILE, 13,
                                       authFile.c_str()),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(QDMI_session_set_parameter(uninitializedSession,
                                       QDMI_SESSION_PARAMETER_CUSTOM1, 0,
                                       nullptr),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(QDMI_session_set_parameter(uninitializedSession,
                                       QDMI_SESSION_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(QDMI_session_set_parameter(session, QDMI_SESSION_PARAMETER_AUTHFILE,
                                       13, authFile.c_str()),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(QDMI_session_set_parameter(nullptr, QDMI_SESSION_PARAMETER_AUTHFILE,
                                       13, authFile.c_str()),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverTest, JobCreate) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(QDMI_device_create_job(device, &job), QDMI_SUCCESS);
  QDMI_job_free(job);
  EXPECT_EQ(QDMI_device_create_job(device, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverTest, JobSetParameter) {
  EXPECT_EQ(QDMI_job_set_parameter(nullptr, QDMI_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverJobTest, JobSetParameter) {
  EXPECT_THAT(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_PROGRAM,
                                     sizeof(QDMI_Program_Format), nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  const QDMI_Program_Format value = QDMI_PROGRAM_FORMAT_QASM2;
  EXPECT_THAT(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_PROGRAMFORMAT,
                                     sizeof(QDMI_Program_Format), &value),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  const size_t numShots = 1;
  EXPECT_THAT(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_SHOTSNUM,
                                     sizeof(size_t), &numShots),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  constexpr std::array customParams{
      QDMI_JOB_PARAMETER_CUSTOM1, QDMI_JOB_PARAMETER_CUSTOM2,
      QDMI_JOB_PARAMETER_CUSTOM3, QDMI_JOB_PARAMETER_CUSTOM4,
      QDMI_JOB_PARAMETER_CUSTOM5,
  };
  for (const auto param : customParams) {
    EXPECT_THAT(QDMI_job_set_parameter(job, param, 0, nullptr),
                testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  }
  EXPECT_EQ(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_MAX, 0, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverTest, JobQueryProperty) {
  EXPECT_EQ(QDMI_job_query_property(nullptr, QDMI_JOB_PROPERTY_MAX, 0, nullptr,
                                    nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverJobTest, JobQueryProperty) {
  EXPECT_THAT(
      QDMI_job_query_property(job, QDMI_JOB_PROPERTY_ID, 0, nullptr, nullptr),
      testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));

  EXPECT_THAT(QDMI_job_query_property(job, QDMI_JOB_PROPERTY_PROGRAM, 0,
                                      nullptr, nullptr),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));

  QDMI_Program_Format value = QDMI_PROGRAM_FORMAT_QASM2;
  auto result = QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_PROGRAMFORMAT,
                                       sizeof(QDMI_Program_Format), &value);
  EXPECT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  if (result == QDMI_SUCCESS) {
    value = QDMI_PROGRAM_FORMAT_MAX;
    EXPECT_EQ(QDMI_job_query_property(job, QDMI_JOB_PROPERTY_PROGRAMFORMAT,
                                      sizeof(QDMI_Program_Format), &value,
                                      nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(value, QDMI_PROGRAM_FORMAT_QASM2);
  }
  size_t numShots = 1;
  result = QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_SHOTSNUM,
                                  sizeof(QDMI_Program_Format), &numShots);
  EXPECT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
  if (result == QDMI_SUCCESS) {
    numShots = 0;
    EXPECT_EQ(QDMI_job_query_property(job, QDMI_JOB_PROPERTY_SHOTSNUM,
                                      sizeof(size_t), &numShots, nullptr),
              QDMI_SUCCESS);
    EXPECT_EQ(numShots, 1);
  }

  EXPECT_EQ(QDMI_job_query_property(job, QDMI_JOB_PROPERTY_QUEUEPOSITION, 0,
                                    nullptr, nullptr),
            QDMI_ERROR_NOTSUPPORTED);

  constexpr std::array customProperties{
      QDMI_JOB_PROPERTY_CUSTOM1, QDMI_JOB_PROPERTY_CUSTOM2,
      QDMI_JOB_PROPERTY_CUSTOM3, QDMI_JOB_PROPERTY_CUSTOM4,
      QDMI_JOB_PROPERTY_CUSTOM5,
  };
  for (const auto property : customProperties) {
    EXPECT_EQ(QDMI_job_query_property(job, property, 0, nullptr, nullptr),
              QDMI_ERROR_NOTSUPPORTED);
  }
}

TEST_P(DriverTest, JobSubmit) {
  EXPECT_EQ(QDMI_job_submit(nullptr), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverJobTest, JobSubmit) {
  EXPECT_THAT(QDMI_job_submit(job),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_P(DriverTest, JobCancel) {
  EXPECT_EQ(QDMI_job_cancel(nullptr), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverJobTest, JobCancel) {
  EXPECT_THAT(QDMI_job_cancel(job),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_INVALIDARGUMENT,
                             QDMI_ERROR_NOTSUPPORTED));
}

TEST_P(DriverTest, JobCheck) {
  EXPECT_EQ(QDMI_job_check(nullptr, nullptr), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverJobTest, JobCheck) {
  QDMI_Job_Status status = QDMI_JOB_STATUS_RUNNING;
  EXPECT_THAT(QDMI_job_check(job, &status),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
}

TEST_P(DriverTest, JobWait) {
  EXPECT_EQ(QDMI_job_wait(nullptr, 0), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverJobTest, JobWait) {
  EXPECT_THAT(QDMI_job_wait(job, 1),
              testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED,
                             QDMI_ERROR_TIMEOUT, QDMI_ERROR_BADSTATE));
}

TEST_P(DriverTest, JobGetResults) {
  EXPECT_EQ(
      QDMI_job_get_results(nullptr, QDMI_JOB_RESULT_MAX, 0, nullptr, nullptr),
      QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverJobTest, JobGetResults) {
  EXPECT_THAT(
      QDMI_job_get_results(job, QDMI_JOB_RESULT_SHOTS, 0, nullptr, nullptr),
      testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED,
                     QDMI_ERROR_BADSTATE));
}

TEST_P(DriverTest, QueryDeviceProperty) {
  EXPECT_EQ(QDMI_device_query_device_property(device, QDMI_DEVICE_PROPERTY_MAX,
                                              0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(QDMI_device_query_device_property(nullptr, QDMI_DEVICE_PROPERTY_MAX,
                                              0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverTest, QuerySiteProperty) {
  EXPECT_EQ(QDMI_device_query_site_property(
                device, nullptr, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(QDMI_device_query_site_property(
                nullptr, nullptr, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverTest, QueryOperationProperty) {
  EXPECT_EQ(QDMI_device_query_operation_property(
                device, nullptr, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(QDMI_device_query_operation_property(
                nullptr, nullptr, 0, nullptr, 0, nullptr,
                QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverTest, QueryDeviceVersion) {
  size_t size = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_VERSION, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a version.";
  std::string value(size - 1, '\0');
  ASSERT_EQ(QDMI_device_query_device_property(device,
                                              QDMI_DEVICE_PROPERTY_VERSION,
                                              size, value.data(), nullptr),
            QDMI_SUCCESS)
      << "Devices must provide a version.";
  EXPECT_FALSE(value.empty()) << "Devices must provide a version.";
}

TEST_P(DriverTest, QueryDeviceLibraryVersion) {
  size_t size = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a library version.";
  std::string value(size - 1, '\0');
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_LIBRARYVERSION, size, value.data(),
                nullptr),
            QDMI_SUCCESS)
      << "Devices must provide a library version.";
  ASSERT_FALSE(value.empty()) << "Devices must provide a library version.";
}

TEST_P(DriverTest, QueryNumQubits) {
  size_t numQubits = 0;
  ASSERT_EQ(
      QDMI_device_query_device_property(device, QDMI_DEVICE_PROPERTY_QUBITSNUM,
                                        sizeof(size_t), &numQubits, nullptr),
      QDMI_SUCCESS)
      << "Devices must provide the number of qubits.";
  EXPECT_GT(numQubits, 0) << "Number of qubits must be greater than 0.";
}

TEST_P(DriverTest, QuerySites) {
  size_t size = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_SITES, 0, nullptr, &size),
            QDMI_SUCCESS)
      << "Devices must provide a list of sites.";
  std::vector<QDMI_Site> sites(size / sizeof(QDMI_Site));
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_SITES, size,
                static_cast<void*>(sites.data()), nullptr),
            QDMI_SUCCESS)
      << "Failed to get sites.";
  std::unordered_set<size_t> ids;
  for (auto* site : sites) {
    uint64_t index = 0;
    EXPECT_EQ(
        QDMI_device_query_site_property(device, site, QDMI_SITE_PROPERTY_INDEX,
                                        sizeof(uint64_t), &index, nullptr),
        QDMI_SUCCESS)
        << "Devices must provide a site id";
    EXPECT_TRUE(ids.emplace(index).second)
        << "Device must provide unique site ids. Found duplicate id: " << index
        << ".";
    double t1 = 0;
    double t2 = 0;
    auto result = QDMI_device_query_site_property(
        device, site, QDMI_SITE_PROPERTY_T1, sizeof(double), &t1, nullptr);
    ASSERT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
    if (result == QDMI_SUCCESS) {
      EXPECT_GT(t1, 0) << "Devices must provide a site T1 time larger than 0.";
    }
    result = QDMI_device_query_site_property(
        device, site, QDMI_SITE_PROPERTY_T2, sizeof(double), &t2, nullptr);
    ASSERT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED));
    if (result == QDMI_SUCCESS) {
      EXPECT_GT(t2, 0) << "Devices must provide a site T2 time larger than 0.";
    }
    EXPECT_EQ(QDMI_device_query_site_property(
                  device, site, QDMI_SITE_PROPERTY_MAX, 0, nullptr, nullptr),
              QDMI_ERROR_INVALIDARGUMENT)
        << "The MAX property is not a valid value for any device.";
  }
}

TEST_P(DriverTest, QueryOperations) {
  size_t operationsSize = 0;
  ASSERT_EQ(QDMI_device_query_device_property(device,
                                              QDMI_DEVICE_PROPERTY_OPERATIONS,
                                              0, nullptr, &operationsSize),
            QDMI_SUCCESS)
      << "Failed to get the size to retrieve the operations.";
  std::vector<QDMI_Operation> operations(operationsSize /
                                         sizeof(QDMI_Operation));
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_OPERATIONS, operationsSize,
                static_cast<void*>(operations.data()), nullptr),
            QDMI_SUCCESS)
      << "Failed to retrieve the operations.";
  for (auto* const op : operations) {
    size_t namesSize = 0;
    ASSERT_EQ(QDMI_device_query_operation_property(
                  device, op, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_NAME, 0, nullptr, &namesSize),
              QDMI_SUCCESS)
        << "Failed to get the length of the operation's name.";
    std::string name(namesSize - 1, '\0');
    ASSERT_EQ(
        QDMI_device_query_operation_property(device, op, 0, nullptr, 0, nullptr,
                                             QDMI_OPERATION_PROPERTY_NAME,
                                             namesSize, name.data(), nullptr),
        QDMI_SUCCESS)
        << "Failed to retrieve the operation's name.";
    EXPECT_FALSE(name.empty())
        << "Device must provide a non-empty name for every operation.";

    size_t numParams = 0;
    ASSERT_EQ(QDMI_device_query_operation_property(
                  device, op, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_PARAMETERSNUM, sizeof(size_t),
                  &numParams, nullptr),
              QDMI_SUCCESS)
        << "Failed to query number of parameters for operation.";

    double duration = 0;
    double fidelity = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::vector<double> params(numParams);
    for (auto& param : params) {
      param = dis(gen);
    }
    auto result = QDMI_device_query_operation_property(
        device, op, 0, nullptr, numParams, params.data(),
        QDMI_OPERATION_PROPERTY_DURATION, sizeof(double), &duration, nullptr);
    ASSERT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED))
        << "Failed to query duration for operation " << name << ".";
    if (result == QDMI_SUCCESS) {
      EXPECT_GT(duration, 0)
          << "Duration must be larger than 0 for operation " << name << ".";
    }
    result = QDMI_device_query_operation_property(
        device, op, 0, nullptr, numParams, params.data(),
        QDMI_OPERATION_PROPERTY_FIDELITY, sizeof(double), &fidelity, nullptr);
    ASSERT_THAT(result, testing::AnyOf(QDMI_SUCCESS, QDMI_ERROR_NOTSUPPORTED))
        << "Failed to query fidelity for operation " << name << ".";
    if (result == QDMI_SUCCESS) {
      EXPECT_THAT(fidelity, testing::IsBetween(0, 1))
          << "Fidelity must be between 0 and 1 for operation " << name << ".";
    }

    EXPECT_EQ(QDMI_device_query_operation_property(
                  device, op, 0, nullptr, 0, nullptr,
                  QDMI_OPERATION_PROPERTY_MAX, 0, nullptr, nullptr),
              QDMI_ERROR_INVALIDARGUMENT)
        << "The MAX property is not a valid value for any device.";
  }
}

TEST_P(DriverTest, SessionAlloc) {
  EXPECT_EQ(QDMI_session_alloc(nullptr), QDMI_ERROR_INVALIDARGUMENT);
}

TEST_P(DriverTest, RetrieveJobById) {
  QDMI_Job job = nullptr;
  EXPECT_EQ(QDMI_session_retrieve_job_by_id(nullptr, "job-id", &job),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(QDMI_session_retrieve_job_by_id(device, nullptr, &job),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(QDMI_session_retrieve_job_by_id(device, "", &job),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(QDMI_session_retrieve_job_by_id(device, "job-id", nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(QDMI_session_retrieve_job_by_id(device, "job-id", &job),
            QDMI_ERROR_NOTSUPPORTED);
  EXPECT_EQ(job, nullptr);
}

TEST_P(DriverTest, SessionInit) {
  EXPECT_EQ(QDMI_session_init(nullptr), QDMI_ERROR_INVALIDARGUMENT)
      << "`session == nullptr` is not a valid argument.";
  EXPECT_EQ(QDMI_session_init(session), QDMI_ERROR_BADSTATE)
      << "Session must return `BADSTATE` if it is initialized again.";
}

TEST_P(DriverTest, QuerySessionProperty) {
  EXPECT_EQ(QDMI_session_query_session_property(
                nullptr, QDMI_SESSION_PROPERTY_DEVICES, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "`session == nullptr` is not a valid argument.";
  EXPECT_EQ(QDMI_session_query_session_property(
                session, QDMI_SESSION_PROPERTY_MAX, 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "`prop >= QDMI_SESSION_PROPERTY_MAX` is not a valid argument.";

  // Must not query on an uninitialized session
  QDMI_Session uninitializedSession = nullptr;
  ASSERT_EQ(QDMI_session_alloc(&uninitializedSession), QDMI_SUCCESS);
  EXPECT_EQ(QDMI_session_query_session_property(uninitializedSession,
                                                QDMI_SESSION_PROPERTY_DEVICES,
                                                0, nullptr, nullptr),
            QDMI_ERROR_BADSTATE);

  constexpr size_t size = sizeof(QDMI_Device) - 1;
  std::array<char, size> devices{};
  EXPECT_EQ(QDMI_session_query_session_property(
                session, QDMI_SESSION_PROPERTY_DEVICES, size,
                static_cast<void*>(devices.data()), nullptr),
            QDMI_ERROR_INVALIDARGUMENT)
      << "Device must return `INVALIDARGUMENT` if the buffer is too small.";
}

TEST_P(DriverTest, QueryNeedsCalibration) {
  size_t needsCalibration = 0;
  const auto ret = QDMI_device_query_device_property(
      device, QDMI_DEVICE_PROPERTY_NEEDSCALIBRATION, sizeof(size_t),
      &needsCalibration, nullptr);
  EXPECT_EQ(ret, QDMI_SUCCESS);
  EXPECT_THAT(needsCalibration, testing::AnyOf(0, 1));
}
constexpr std::array DEVICES{
    "MQT SC Default QDMI Device",
#ifdef MQT_CORE_QDMI_HAS_DDSIM_DEVICE
    "MQT Core DDSIM QDMI Device",
#endif
};

namespace {
void registerSessionTestDevice() {
  static_cast<void>(qdmi::Driver::get().registerDeviceIfAbsent({
      .id = "test.session-overrides",
      .library = MQT_CORE_QDMI_SESSION_DEVICE,
      .prefix = "TEST_SESSION",
      .session =
          {
              .baseUrl = "registered-base",
              .token = "registered-token",
              .custom1 = "registered-custom",
          },
  }));
}
} // namespace

// Instantiate the test suite with different parameters
INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    DefaultDevices,
    // Test suite name
    DriverTest,
    // Parameters to test with
    testing::ValuesIn(DEVICES),
    [](const testing::TestParamInfo<const char*>& paramInfo) {
      std::string name = paramInfo.param;
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      // Remove parentheses for valid test names
      std::erase(name, '(');
      std::erase(name, ')');
      return name;
    });

TEST(ConfiguredDriverTest, ConstructionRegistersWithoutOpeningDevices) {
  const auto [library, prefix] = TEST_DEVICE_LIBRARIES.front();
  EXPECT_NO_THROW(qdmi::Driver::get().registerDevice(
      {.id = "mqt.sc.default", .library = library, .prefix = prefix}, true));
}

TEST(ConfiguredDriverTest, ExposesWorkingDefinitionsAndIsolatesFailures) {
  QDMI_Session session = nullptr;
  ASSERT_EQ(QDMI_session_alloc(&session), QDMI_SUCCESS);
  ASSERT_EQ(QDMI_session_init(session), QDMI_SUCCESS);

  size_t size = 0;
  ASSERT_EQ(QDMI_session_query_session_property(
                session, QDMI_SESSION_PROPERTY_DEVICES, 0, nullptr, &size),
            QDMI_SUCCESS);
  ASSERT_EQ(size % sizeof(QDMI_Device), 0);
  std::vector<QDMI_Device> devices(size / sizeof(QDMI_Device));
  ASSERT_EQ(QDMI_session_query_session_property(
                session, QDMI_SESSION_PROPERTY_DEVICES, size,
                static_cast<void*>(devices.data()), nullptr),
            QDMI_SUCCESS);

  std::vector<std::string> ids;
  std::ranges::transform(devices, std::back_inserter(ids), [](auto device) {
    return queryTextProperty(device, QDMI_DEVICE_PROPERTY_ID);
  });
  std::vector<std::string> expectedIds{
      "mqt.sc.default",          "mqt.sc.iqm.emerald",
      "mqt.sc.iqm.garnet",       "test.sc.runtime-default",
      "test.sc.runtime-custom",  "test.sc.runtime-one",
      "test.sc.runtime-two",     "test.session-overrides",
      "test.session-with-child", "test.typed-configuration",
  };
#ifdef MQT_CORE_QDMI_HAS_DDSIM_DEVICE
  expectedIds.emplace_back("mqt.ddsim.default");
#endif
  EXPECT_THAT(ids, testing::UnorderedElementsAreArray(expectedIds));
  QDMI_session_free(session);
}

TEST(ConfiguredDriverTest,
     ExtensionAbiValidatesArgumentsAndPropagatesDeviceStatus) {
  EXPECT_EQ(MQT_CORE_QDMI_driver_add_manifest_v1(nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_CORE_QDMI_driver_add_manifest_v1(""),
            QDMI_ERROR_INVALIDARGUMENT);

  QDMI_Session session = nullptr;
  EXPECT_EQ(MQT_CORE_QDMI_driver_session_alloc_for_device_v1(
                "test.session-overrides", 0, nullptr, nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_CORE_QDMI_driver_session_alloc_for_device_v1(nullptr, 0,
                                                             nullptr, &session),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_CORE_QDMI_driver_session_alloc_for_device_v1("", 0, nullptr,
                                                             &session),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_CORE_QDMI_driver_session_alloc_for_device_v1(
                "test.session-overrides", 1, nullptr, &session),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(MQT_CORE_QDMI_driver_session_alloc_for_device_v1("test.disabled", 0,
                                                             nullptr, &session),
            QDMI_ERROR_PERMISSIONDENIED);
  EXPECT_EQ(session, nullptr);
  EXPECT_EQ(MQT_CORE_QDMI_driver_session_alloc_for_device_v1(
                "test.unknown-target", 0, nullptr, &session),
            QDMI_ERROR_NOTFOUND);
  EXPECT_EQ(session, nullptr);
}

TEST(DeviceRegistrationTest, ValidatesDuplicatesAndReplacement) {
  auto& driver = qdmi::Driver::get();
  EXPECT_THROW(driver.registerDevice({}), std::invalid_argument);
  EXPECT_THROW(driver.open("test.unknown"), std::out_of_range);

  const auto [library, prefix] = TEST_DEVICE_LIBRARIES.front();
  const qdmi::DeviceDefinition original{
      .id = "test.replaceable",
      .library = library,
      .prefix = prefix,
  };
  auto invalidId = original;
  invalidId.id = std::string{"test.alias\0hidden", 17};
  EXPECT_THROW(driver.registerDevice(std::move(invalidId)),
               std::invalid_argument);
  auto invalidLibrary = original;
  invalidLibrary.library =
      std::filesystem::path{std::string{"device\0alias", 12}};
  EXPECT_THROW(driver.registerDevice(std::move(invalidLibrary)),
               std::invalid_argument);
  auto invalidPrefix = original;
  invalidPrefix.prefix.clear();
  EXPECT_THROW(driver.registerDevice(std::move(invalidPrefix)),
               std::invalid_argument);
  driver.registerDevice(original);
  EXPECT_THROW(driver.registerDevice(original), std::invalid_argument);

  auto replacement = original;
  replacement.session.custom3 = "replacement";
  EXPECT_NO_THROW(driver.registerDevice(replacement, true));
  auto* const opened = driver.open(original.id);
  ASSERT_NE(opened, nullptr);
  EXPECT_EQ(driver.open(original.id), opened);
  EXPECT_THROW(driver.registerDevice(original, true), std::runtime_error);
  EXPECT_NO_THROW(driver.registerDevice(
      {.id = "test.upserted", .library = library, .prefix = prefix}, true));
  EXPECT_NE(driver.open("test.upserted"), nullptr);
}

TEST(DeviceRegistrationTest, RegistersOnlyWhenIdIsAbsent) {
  auto& driver = qdmi::Driver::get();
  const auto [library, prefix] = TEST_DEVICE_LIBRARIES.front();
  const qdmi::DeviceDefinition definition{
      .id = "test.insert-if-absent",
      .library = library,
      .prefix = prefix,
  };
  EXPECT_TRUE(driver.registerDeviceIfAbsent(definition));
  EXPECT_FALSE(driver.registerDeviceIfAbsent(definition));

  auto invalidDuplicate = definition;
  invalidDuplicate.library.clear();
  EXPECT_THROW(static_cast<void>(
                   driver.registerDeviceIfAbsent(std::move(invalidDuplicate))),
               std::invalid_argument);

  const qdmi::DeviceDefinition disabled{
      .id = "test.disabled",
      .library = library,
      .prefix = prefix,
  };
  EXPECT_FALSE(driver.registerDeviceIfAbsent(disabled));
  EXPECT_THROW(static_cast<void>(driver.open(disabled.id)), std::runtime_error);
  EXPECT_THROW(driver.registerDevice(disabled), std::invalid_argument);
}

TEST(DeviceRegistrationTest, ConcurrentRegistrationInsertsOnce) {
  constexpr size_t threadCount = 8;
  auto& driver = qdmi::Driver::get();
  const auto [library, prefix] = TEST_DEVICE_LIBRARIES.front();
  const qdmi::DeviceDefinition definition{
      .id = "test.concurrent-insert-if-absent",
      .library = library,
      .prefix = prefix,
  };
  std::array<bool, threadCount> inserted{};
  std::barrier start(threadCount);
  std::vector<std::thread> threads;
  threads.reserve(threadCount);

  for (size_t thread = 0; thread < threadCount; ++thread) {
    threads.emplace_back([&, thread] {
      start.arrive_and_wait();
      inserted[thread] = driver.registerDeviceIfAbsent(definition);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(std::ranges::count(inserted, true), 1);
  EXPECT_EQ(std::ranges::count(driver.registeredDeviceIds(), definition.id), 1);
}

TEST(DeviceRegistrationTest, ConcurrentOpenReturnsOnePersistentDevice) {
  constexpr size_t threadCount = 8;
  auto& driver = qdmi::Driver::get();
  const auto [library, prefix] = TEST_DEVICE_LIBRARIES.front();
  const qdmi::DeviceDefinition definition{
      .id = "test.concurrent-open",
      .library = library,
      .prefix = prefix,
  };
  driver.registerDevice(definition);
  std::array<QDMI_Device, threadCount> devices{};
  std::barrier start(threadCount);
  std::vector<std::thread> threads;
  threads.reserve(threadCount);

  for (size_t thread = 0; thread < threadCount; ++thread) {
    threads.emplace_back([&, thread] {
      start.arrive_and_wait();
      devices[thread] = driver.open(definition.id);
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  ASSERT_NE(devices.front(), nullptr);
  EXPECT_TRUE(std::ranges::all_of(
      devices, [&](const auto device) { return device == devices.front(); }));
}

TEST(DeviceRegistrationTest, RegistrationDoesNotLoadLibraries) {
  auto& driver = qdmi::Driver::get();
  driver.registerDevice({
      .id = "test.missing-library",
      .library = "/nonexistent/device-library",
      .prefix = "MISSING",
  });
  EXPECT_THROW(static_cast<void>(driver.open("test.missing-library")),
               std::runtime_error);
}

TEST(DeviceRegistrationTest,
     EnumeratesEnabledIdsInOrderWithoutLoadingLibraries) {
  auto& driver = qdmi::Driver::get();
  const auto idsBefore = driver.registeredDeviceIds();
  driver.registerDevice({
      .id = "test.enumeration.first",
      .library = "/nonexistent/first-device-library",
      .prefix = "MISSING_FIRST",
  });
  driver.registerDevice({
      .id = "test.enumeration.second",
      .library = "/nonexistent/second-device-library",
      .prefix = "MISSING_SECOND",
  });

  const auto idsAfter = driver.registeredDeviceIds();
  ASSERT_EQ(idsAfter.size(), idsBefore.size() + 2);
  EXPECT_TRUE(std::equal(idsBefore.begin(), idsBefore.end(), idsAfter.begin()));
  EXPECT_EQ(idsAfter[idsBefore.size()], "test.enumeration.first");
  EXPECT_EQ(idsAfter[idsBefore.size() + 1], "test.enumeration.second");
  EXPECT_THAT(idsAfter, testing::Not(testing::Contains("test.disabled")));

  EXPECT_THROW(static_cast<void>(driver.open("test.enumeration.first")),
               std::runtime_error);
}

TEST(DeviceRegistrationTest, SynthesizesManifestForMetadataOnlyTarget) {
  std::ifstream manifest(MQT_CORE_QDMI_METADATA_MANIFEST);
  ASSERT_TRUE(manifest);
  const std::string contents{std::istreambuf_iterator<char>(manifest),
                             std::istreambuf_iterator<char>()};
  EXPECT_THAT(contents, testing::HasSubstr("\"id\": \"test.metadata-only\""));
  EXPECT_THAT(contents, testing::HasSubstr("\"prefix\": \"TEST_METADATA\""));
  EXPECT_THAT(contents, testing::HasSubstr("mqt-core-qdmi-metadata-device"));
}

TEST(DeviceRegistrationTest, TypedConfigurationUsesExactlyOneAdapterSlot) {
  static_cast<void>(qdmi::Driver::get().registerDeviceIfAbsent({
      .id = "test.typed-configuration",
      .library = MQT_CORE_QDMI_SESSION_DEVICE,
      .prefix = "TEST_SESSION",
      .session =
          {
              .deviceConfiguration =
                  qdmi::InlineDeviceConfiguration{
                      .json = R"({"name":"inline"})",
                  },
          },
  }));

  const auto inlineDevice =
      qdmi::Session::openDevice("test.typed-configuration");
  EXPECT_THAT(
      inlineDevice.getName(),
      testing::HasSubstr(R"(custom1={"name":"inline"};custom2=<unset>)"));
}

TEST(DeviceRegistrationTest, TypedConfigurationRejectsRawAdapterSlotConflict) {
  auto& driver = qdmi::Driver::get();
  const qdmi::DeviceDefinition definition{
      .id = "test.typed-conflict",
      .library = MQT_CORE_QDMI_SESSION_DEVICE,
      .prefix = "TEST_SESSION",
      .session =
          {
              .deviceConfiguration =
                  qdmi::InlineDeviceConfiguration{.json = "{}"},
              .custom1 = "raw",
          },
  };
  EXPECT_THROW(driver.registerDevice(definition), std::invalid_argument);
}

TEST(DeviceRegistrationTest,
     RuntimeConfigurationSeparatesModelsUsingOneScProviderLibrary) {
  auto& driver = qdmi::Driver::get();
  static_cast<void>(driver.registerDeviceIfAbsent({
      .id = "test.sc.runtime-default",
      .library = MQT_CORE_QDMI_SC_LIBRARY,
      .prefix = "MQT_SC",
  }));
  static_cast<void>(driver.registerDeviceIfAbsent({
      .id = "test.sc.runtime-custom",
      .library = MQT_CORE_QDMI_SC_LIBRARY,
      .prefix = "MQT_SC",
      .session =
          {
              .deviceConfiguration =
                  qdmi::FileDeviceConfiguration{
                      .path = MQT_CORE_QDMI_CUSTOM_SC_FILE,
                  },
          },
  }));

  const auto defaultDevice =
      qdmi::Session::openDevice("test.sc.runtime-default");
  const auto customDevice = qdmi::Session::openDevice("test.sc.runtime-custom");
  EXPECT_EQ(defaultDevice.getName(), "MQT SC Default QDMI Device");
  EXPECT_EQ(customDevice.getName(), "Custom SC Driver Device");
  ASSERT_FALSE(defaultDevice.getOperations().empty());
  ASSERT_FALSE(customDevice.getOperations().empty());
  EXPECT_EQ(defaultDevice.getOperations().front().getDuration(), 20);
  EXPECT_EQ(customDevice.getOperations().front().getDuration(), 77);

  static_cast<void>(driver.registerDeviceIfAbsent({
      .id = "test.sc.runtime-invalid",
      .library = MQT_CORE_QDMI_SC_LIBRARY,
      .prefix = "MQT_SC",
      .session =
          {
              .deviceConfiguration =
                  qdmi::InlineDeviceConfiguration{
                      .json = "{}",
                  },
          },
  }));
  EXPECT_THROW(
      static_cast<void>(qdmi::Session::openDevice("test.sc.runtime-invalid")),
      std::out_of_range);
  EXPECT_EQ(qdmi::Session::openDevice("test.sc.runtime-default").getName(),
            "MQT SC Default QDMI Device");
}

TEST(DeviceRegistrationTest, FreshOpenCreatesDistinctSessions) {
  registerSessionTestDevice();
  const auto first = qdmi::Session::openDevice("test.session-overrides");
  const auto second = qdmi::Session::openDevice("test.session-overrides");
  EXPECT_NE(first, second);
}

TEST(DeviceRegistrationTest, CustomOperationListSupportsRawAndQDMIQueries) {
  registerSessionTestDevice();
  const auto device = qdmi::Session::openDevice("test.session-overrides");

  size_t size = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_CUSTOM1, 0, nullptr, &size),
            QDMI_SUCCESS);
  ASSERT_EQ(size, 2 * sizeof(QDMI_Operation));
  std::vector<QDMI_Operation> rawOperations(size / sizeof(QDMI_Operation));
  EXPECT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_CUSTOM1,
                size - sizeof(QDMI_Operation),
                static_cast<void*>(rawOperations.data()), nullptr),
            QDMI_ERROR_INVALIDARGUMENT);
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_CUSTOM1, size,
                static_cast<void*>(rawOperations.data()), nullptr),
            QDMI_SUCCESS);

  const auto operations =
      device.queryCustomOperations(qdmi::CustomProperty::Custom1);
  ASSERT_TRUE(operations.has_value());
  ASSERT_EQ(operations->size(), rawOperations.size());
  EXPECT_EQ(static_cast<QDMI_Operation>(operations->at(0)),
            rawOperations.at(0));
  EXPECT_EQ(static_cast<QDMI_Operation>(operations->at(1)),
            rawOperations.at(1));
  EXPECT_EQ(operations->at(0).getName(), "custom-rx");
  EXPECT_EQ(operations->at(0).getQubitsNum(), 1);
  EXPECT_EQ(operations->at(0).getParametersNum(), 1);
  EXPECT_EQ(operations->at(1).getName(), "custom-cx");
  EXPECT_EQ(operations->at(1).getQubitsNum(), 2);
  EXPECT_EQ(operations->at(1).getParametersNum(), 0);
}

TEST(DeviceRegistrationTest,
     CustomOperationListDistinguishesUnsupportedEmptyAndMalformed) {
  registerSessionTestDevice();
  const auto device = qdmi::Session::openDevice("test.session-overrides");

  EXPECT_EQ(device.queryCustomOperations(qdmi::CustomProperty::Custom4),
            std::nullopt);
  const auto empty =
      device.queryCustomOperations(qdmi::CustomProperty::Custom2);
  ASSERT_TRUE(empty.has_value());
  EXPECT_TRUE(empty->empty());
  EXPECT_THROW(static_cast<void>(
                   device.queryCustomOperations(qdmi::CustomProperty::Custom3)),
               std::invalid_argument);
}

TEST(DeviceRegistrationTest, CustomOperationRetainsOwningDeviceSession) {
  registerSessionTestDevice();
  const auto operation = [] {
    const auto operations =
        qdmi::Session::openDevice("test.session-overrides")
            .queryCustomOperations(qdmi::CustomProperty::Custom1);
    if (!operations.has_value() || operations->empty()) {
      throw std::runtime_error("Test provider returned no custom operations");
    }
    return operations->front();
  }();

  EXPECT_EQ(operation.getName(), "custom-rx");
  EXPECT_EQ(operation.getQubitsNum(), 1);
}

TEST(DeviceRegistrationTest,
     ScRuntimeConfigurationSeparatesModelsUsingOneProviderLibrary) {
  auto& driver = qdmi::Driver::get();
  static_cast<void>(driver.registerDeviceIfAbsent({
      .id = "test.sc.runtime-one",
      .library = MQT_CORE_QDMI_SC_LIBRARY,
      .prefix = "MQT_SC",
      .session =
          {
              .deviceConfiguration =
                  qdmi::InlineDeviceConfiguration{
                      .json = R"({
                 "schema-version":1,
                 "name":"SC runtime one",
                 "numQubits":1,
                 "durationUnit":{"unit":"ns","scaleFactor":1},
                 "qubitProperties":{"defaults":{"t1":10,"t2":20},"overrides":[]},
                 "couplings":[],
                 "operations":[{
                   "name":"r",
                   "numParameters":2,
                   "numQubits":1,
                   "duration":7,
                   "fidelity":0.8
                 }]
               })",
                  },
          },
  }));
  static_cast<void>(driver.registerDeviceIfAbsent({
      .id = "test.sc.runtime-two",
      .library = MQT_CORE_QDMI_SC_LIBRARY,
      .prefix = "MQT_SC",
      .session =
          {
              .deviceConfiguration =
                  qdmi::InlineDeviceConfiguration{
                      .json = R"({
                 "schema-version":1,
                 "name":"SC runtime two",
                 "numQubits":2,
                 "durationUnit":{"unit":"us","scaleFactor":0.5},
                 "qubitProperties":{"defaults":{},"overrides":[]},
                 "couplings":[[1,0]],
                 "operations":[]
               })",
                  },
          },
  }));

  const auto first = qdmi::Session::openDevice("test.sc.runtime-one");
  const auto second = qdmi::Session::openDevice("test.sc.runtime-two");
  EXPECT_EQ(first.getName(), "SC runtime one");
  EXPECT_EQ(first.getQubitsNum(), 1);
  const auto firstSites = first.getSites();
  ASSERT_EQ(firstSites.size(), 1);
  EXPECT_EQ(firstSites[0].getT1(), 10);
  EXPECT_EQ(firstSites[0].getT2(), 20);
  const auto firstOperations = first.getOperations();
  ASSERT_EQ(firstOperations.size(), 1);
  EXPECT_EQ(firstOperations[0].getName(), "r");
  EXPECT_EQ(firstOperations[0].getDuration({firstSites[0]}), 7);
  EXPECT_EQ(firstOperations[0].getFidelity({firstSites[0]}), 0.8);
  EXPECT_EQ(second.getName(), "SC runtime two");
  EXPECT_EQ(second.getQubitsNum(), 2);
  const auto secondCouplingMap = second.getCouplingMap();
  ASSERT_TRUE(secondCouplingMap.has_value());
  ASSERT_EQ(secondCouplingMap->size(), 1);
}

TEST(DeviceRegistrationTest, FreshJobRetainsItsDeviceSession) {
  registerSessionTestDevice();
  std::optional<qdmi::Job> job;
  std::string baseline;
  {
    auto const device = qdmi::Session::openDevice("test.session-overrides");
    baseline = device.getName();
    job.emplace(
        device.submitJob("OPENQASM 2.0;", QDMI_PROGRAM_FORMAT_QASM2, 1));
  }

  ASSERT_TRUE(job.has_value());
  EXPECT_EQ(job->getId(), "session-job");
  job.reset();

  const auto probe = qdmi::Session::openDevice("test.session-overrides");
  EXPECT_EQ(queryName(probe), baseline);
}

TEST(DeviceRegistrationTest, CustomBinaryJobDoesNotRequireShots) {
  registerSessionTestDevice();
  const auto device = qdmi::Session::openDevice("test.session-overrides");
  constexpr std::array payload{std::byte{0}, std::byte{1}};

  EXPECT_NO_THROW(std::ignore =
                      device.submitJob(payload, QDMI_PROGRAM_FORMAT_CUSTOM1));
  EXPECT_THROW(std::ignore = device.submitJob(
                   payload, QDMI_PROGRAM_FORMAT_CUSTOM1, size_t{1}),
               std::runtime_error);
}

TEST(DeviceRegistrationTest, TextJobDoesNotRequireShots) {
  registerSessionTestDevice();
  const auto device = qdmi::Session::openDevice("test.session-overrides");

  EXPECT_NO_THROW(std::ignore = device.submitJob("OPENQASM 2.0;",
                                                 QDMI_PROGRAM_FORMAT_QASM2));
  EXPECT_THROW(std::ignore = device.submitJob(std::string{"binary"},
                                              QDMI_PROGRAM_FORMAT_QPY),
               std::invalid_argument);
}

TEST(DeviceRegistrationTest, ConcurrentlyFreesDistinctJobs) {
  constexpr size_t threadCount = 8;
  constexpr size_t jobCount = 64;
  registerSessionTestDevice();
  const auto device = qdmi::Session::openDevice("test.session-overrides");
  size_t freedBefore = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_CUSTOM5, sizeof(freedBefore),
                &freedBefore, nullptr),
            QDMI_SUCCESS);
  std::array<QDMI_Job, jobCount> jobs{};
  for (auto& job : jobs) {
    ASSERT_EQ(QDMI_device_create_job(device, &job), QDMI_SUCCESS);
  }

  std::barrier start(threadCount);
  std::vector<std::thread> threads;
  threads.reserve(threadCount);
  for (size_t thread = 0; thread < threadCount; ++thread) {
    threads.emplace_back([&, thread] {
      start.arrive_and_wait();
      for (size_t job = thread; job < jobCount; job += threadCount) {
        QDMI_job_free(jobs[job]);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  size_t freedAfter = 0;
  ASSERT_EQ(QDMI_device_query_device_property(
                device, QDMI_DEVICE_PROPERTY_CUSTOM5, sizeof(freedAfter),
                &freedAfter, nullptr),
            QDMI_SUCCESS);
  EXPECT_EQ(freedAfter - freedBefore, jobCount);
}

TEST(DeviceRegistrationTest, RetrievesExistingJobs) {
  registerSessionTestDevice();
  const auto device = qdmi::Session::openDevice("test.session-overrides");

  QDMI_Job job = nullptr;
  ASSERT_EQ(QDMI_session_retrieve_job_by_id(device, "session-job", &job),
            QDMI_SUCCESS);
  ASSERT_NE(job, nullptr);

  std::array<char, 12> id{};
  EXPECT_EQ(QDMI_job_query_property(job, QDMI_JOB_PROPERTY_ID, id.size(),
                                    id.data(), nullptr),
            QDMI_SUCCESS);
  EXPECT_STREQ(id.data(), "session-job");
  EXPECT_EQ(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_SHOTSNUM,
                                   sizeof(size_t), nullptr),
            QDMI_ERROR_BADSTATE);
  EXPECT_EQ(QDMI_job_submit(job), QDMI_ERROR_BADSTATE);
  QDMI_job_free(job);

  job = nullptr;
  EXPECT_EQ(QDMI_session_retrieve_job_by_id(device, "missing", &job),
            QDMI_ERROR_NOTFOUND);
  EXPECT_EQ(job, nullptr);

  const auto retrievedJob = device.retrieveJobById("session-job");
  EXPECT_EQ(retrievedJob.getId(), "session-job");
}

TEST(DeviceRegistrationTest, RuntimeRegistrationsStayOutOfClientCatalog) {
  auto& driver = qdmi::Driver::get();
  QDMI_Session existingSession = nullptr;
  ASSERT_EQ(QDMI_session_alloc(&existingSession), QDMI_SUCCESS);
  ASSERT_EQ(QDMI_session_init(existingSession), QDMI_SUCCESS);
  size_t originalSize = 0;
  ASSERT_EQ(QDMI_session_query_session_property(existingSession,
                                                QDMI_SESSION_PROPERTY_DEVICES,
                                                0, nullptr, &originalSize),
            QDMI_SUCCESS);

  const auto [library, prefix] = TEST_DEVICE_LIBRARIES.front();
  driver.registerDevice(
      {.id = "test.snapshot", .library = library, .prefix = prefix});
  ASSERT_NE(driver.open("test.snapshot"), nullptr);

  size_t existingSize = 0;
  EXPECT_EQ(QDMI_session_query_session_property(existingSession,
                                                QDMI_SESSION_PROPERTY_DEVICES,
                                                0, nullptr, &existingSize),
            QDMI_SUCCESS);
  EXPECT_EQ(existingSize, originalSize);

  QDMI_Session newSession = nullptr;
  ASSERT_EQ(QDMI_session_alloc(&newSession), QDMI_SUCCESS);
  ASSERT_EQ(QDMI_session_init(newSession), QDMI_SUCCESS);
  size_t newSize = 0;
  EXPECT_EQ(QDMI_session_query_session_property(newSession,
                                                QDMI_SESSION_PROPERTY_DEVICES,
                                                0, nullptr, &newSize),
            QDMI_SUCCESS);
  EXPECT_EQ(newSize, originalSize);
  QDMI_session_free(newSession);
  QDMI_session_free(existingSession);
}

TEST(DeviceSessionConfigTest, OpenWithBaseUrl) {
  qdmi::DeviceSessionConfig config;
  config.baseUrl = "http://localhost:8080";

  for (const auto& [lib, prefix] : TEST_DEVICE_LIBRARIES) {
    EXPECT_NO_THROW(
        { static_cast<void>(openTestDevice(lib, prefix, config)); });
  }
}

TEST(DeviceSessionConfigTest, ReportsSkippedUnsupportedParameter) {
  const auto [library, prefix] = TEST_DEVICE_LIBRARIES.front();
  qdmi::DeviceSessionConfig config;
  config.baseUrl = "http://localhost:8080";

  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(
      { static_cast<void>(openTestDevice(library, prefix, config)); });
  const auto diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_THAT(
      diagnostic,
      testing::AllOf(
          testing::HasSubstr("[mqt-core] [info]"),
          testing::HasSubstr("Device session parameter BASE URL not supported "
                             "by device (skipped)")));
}

TEST(DeviceSessionConfigTest, OpenWithCustomParameters) {
  qdmi::DeviceSessionConfig config;
  config.custom3 = "RESONANCE_COCOS_V1";
  config.custom4 = "test_value";
  config.baseUrl = "http://localhost:9090";

  for (const auto& [lib, prefix] : TEST_DEVICE_LIBRARIES) {
    // Custom parameters may fail with validation errors or succeed/return false
    try {
      static_cast<void>(openTestDevice(lib, prefix, config));
      SUCCEED() << "Library loaded or already loaded";
    } catch (const std::runtime_error& e) {
      // Custom parameters may be rejected with INVALIDARGUMENT
      const std::string msg = e.what();
      if (msg.find("CUSTOM") != std::string::npos &&
          msg.find("Invalid argument") != std::string::npos) {
        SUCCEED() << "Custom parameter validation error (expected): " << msg;
      } else {
        throw; // Re-throw unexpected errors
      }
    }
  }
}

TEST(DeviceSessionConfigTest, OpenWithAuthToken) {
  qdmi::DeviceSessionConfig config;
  config.token = "test_token_123";
  config.baseUrl = "https://api.example.com";

  for (const auto& [lib, prefix] : TEST_DEVICE_LIBRARIES) {
    EXPECT_NO_THROW(
        { static_cast<void>(openTestDevice(lib, prefix, config)); });
  }
}

TEST(DeviceSessionConfigTest, OpenWithAuthFile) {
  qdmi::DeviceSessionConfig config;
  config.authFile = "/nonexistent/auth.json";

  for (const auto& [lib, prefix] : TEST_DEVICE_LIBRARIES) {
    // This should not throw even with non-existent file because
    // if the auth file parameter is not supported, it's skipped
    EXPECT_NO_THROW(
        { static_cast<void>(openTestDevice(lib, prefix, config)); });
  }
}

TEST(DeviceSessionConfigTest, OpenWithUsernamePassword) {
  qdmi::DeviceSessionConfig config;
  config.authUrl = "https://auth.example.com";
  config.username = "quantum_user";
  config.password = "secret_password";

  for (const auto& [lib, prefix] : TEST_DEVICE_LIBRARIES) {
    EXPECT_NO_THROW(
        { static_cast<void>(openTestDevice(lib, prefix, config)); });
  }
}

TEST(DeviceSessionConfigTest,
     OpenWithAuthenticationAndRemainingCustomParameters) {
  qdmi::DeviceSessionConfig config;
  config.baseUrl = "http://localhost:8080";
  config.token = "test_token";
  config.authUrl = "https://auth.example.com";
  config.username = "user";
  config.password = "pass";
  config.custom3 = "value3";
  config.custom4 = "value4";
  config.custom5 = "value5";

  for (const auto& [lib, prefix] : TEST_DEVICE_LIBRARIES) {
    try {
      static_cast<void>(openTestDevice(lib, prefix, config));
      SUCCEED() << "Library loaded or already loaded";
    } catch (const std::runtime_error& e) {
      // Custom parameters may be rejected with INVALIDARGUMENT
      const std::string msg = e.what();
      if (msg.find("CUSTOM") != std::string::npos &&
          msg.find("Invalid argument") != std::string::npos) {
        SUCCEED() << "Custom parameter validation error (expected): " << msg;
      } else {
        throw; // Re-throw unexpected errors
      }
    }
  }
}

TEST(DeviceSessionConfigTest, IdempotentLoadingWithDifferentConfigs) {
  // This test is explicitly not part of the fixture because this would
  // automatically load the default config and the respective libraries.
  if constexpr (TEST_DEVICE_LIBRARIES.empty()) {
    GTEST_SKIP() << "No dynamic device libraries to test";
  }
  for (const auto& [lib, prefix] : TEST_DEVICE_LIBRARIES) {
    // Config 1: baseUrl
    {
      qdmi::DeviceSessionConfig config;
      config.baseUrl = "http://localhost:8080";
      EXPECT_NO_THROW(static_cast<void>(openTestDevice(lib, prefix, config)););
    }

    // Config 2: different baseUrl and custom parameters
    {
      qdmi::DeviceSessionConfig config;
      config.baseUrl = "http://localhost:9090";
      config.custom3 = "API_V2";
      EXPECT_NO_THROW(static_cast<void>(openTestDevice(lib, prefix, config)););
    }

    // Config 3: authentication parameters
    {
      qdmi::DeviceSessionConfig config;
      config.token = "new_token";
      config.authUrl = "https://auth.example.com";
      EXPECT_NO_THROW(static_cast<void>(openTestDevice(lib, prefix, config)););
    }
  }
}

TEST(DynamicDeviceLibraryTest, ReusesLibraryWithFreshDeviceSessions) {
  const auto [library, prefix] = TEST_DEVICE_LIBRARIES.front();
  auto* const first =
      openTestDevice(library, prefix, {.custom3 = "first-session"});
  const auto equivalentLibrary = std::filesystem::path(library).parent_path() /
                                 "." /
                                 std::filesystem::path(library).filename();
  auto* const second = openTestDevice(equivalentLibrary.string(), prefix,
                                      {.custom3 = "second-session"});

  ASSERT_NE(first, second);
  EXPECT_EQ(&first->getLibrary(), &second->getLibrary());
}

TEST(DynamicDeviceLibraryTest, OpenReturnsDevice) {
  if constexpr (TEST_DEVICE_LIBRARIES.empty()) {
    GTEST_SKIP() << "No dynamic device libraries configured for testing.";
  }
  for (const auto& [lib, prefix] : TEST_DEVICE_LIBRARIES) {
    const qdmi::DeviceSessionConfig config;
    QDMI_Device device = nullptr;
    ASSERT_NO_THROW({ device = openTestDevice(lib, prefix, config); });
    ASSERT_NE(device, nullptr)
        << "open should return a non-null device pointer";

    // Verify the device is valid by querying its name
    size_t size = 0;
    EXPECT_EQ(QDMI_device_query_device_property(
                  device, QDMI_DEVICE_PROPERTY_NAME, 0, nullptr, &size),
              QDMI_SUCCESS);
    EXPECT_GT(size, 0) << "Device should have a non-empty name";
  }
}

INSTANTIATE_TEST_SUITE_P(
    // Custom instantiation name
    DefaultDevices,
    // Test suite name
    DriverJobTest,
    // Parameters to test with
    testing::ValuesIn(DEVICES),
    [](const testing::TestParamInfo<const char*>& paramInfo) {
      std::string name = paramInfo.param;
      // Replace spaces with underscores for valid test names
      std::ranges::replace(name, ' ', '_');
      // Remove parentheses for valid test names
      std::erase(name, '(');
      std::erase(name, ')');
      return name;
    });
} // namespace
