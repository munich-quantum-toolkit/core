/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "DeviceApi.hpp"
#include "DeviceState.hpp"
#include "qdmi/Device.hpp"
#include "qdmi/DeviceManager.hpp"

#include <gtest/gtest.h>
#include <qdmi/device.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdmi::detail {
namespace {
// The fake implements a C ABI whose opaque handles intentionally require
// ownership and pointer casts at this isolated test boundary.
// NOLINTBEGIN(readability-named-parameter,cppcoreguidelines-owning-memory)
class ScriptedDeviceApi final {
  struct Child {
    size_t id;
  };
  struct Session {
    const ScriptedDeviceApi* owner = nullptr;
    QDMI_Child_Device child = nullptr;
  };
  struct ScriptedJob {
    const ScriptedDeviceApi* owner = nullptr;
  };

  mutable std::array<Child, 2> children_{{{0}, {1}}};
  mutable ScriptedJob job_;
  static thread_local const ScriptedDeviceApi* activeApi;

  [[nodiscard]] static auto asSession(QDMI_Device_Session session) -> Session* {
    return reinterpret_cast<Session*>(session);
  }
  [[nodiscard]] static auto asJob(QDMI_Device_Job job) -> ScriptedJob* {
    return reinterpret_cast<ScriptedJob*>(job);
  }

public:
  enum class ChildBehavior : std::uint8_t {
    Supported,
    Unsupported,
    Malformed,
    QueryFailure,
    SelectionFailure,
  };

  mutable ChildBehavior behavior = ChildBehavior::Supported;
  mutable size_t opened = 0;
  mutable size_t closed = 0;
  mutable std::vector<std::string> closeOrder;
  mutable QDMI_Device_Status deviceStatus = QDMI_DEVICE_STATUS_IDLE;
  mutable QDMI_Job_Status jobStatus = QDMI_JOB_STATUS_CREATED;
  mutable QDMI_Program_Format programFormat = QDMI_PROGRAM_FORMAT_QASM3;

  [[nodiscard]] auto deviceApi() const -> std::shared_ptr<const DeviceApi> {
    activeApi = this;
    auto api = std::make_shared<DeviceApi>();
    api->device_session_alloc = [](QDMI_Device_Session* session) -> int {
      ++activeApi->opened;
      *session = reinterpret_cast<QDMI_Device_Session>(
          new Session{.owner = activeApi});
      return QDMI_SUCCESS;
    };
    api->device_session_init = [](QDMI_Device_Session) -> int {
      return QDMI_SUCCESS;
    };
    api->device_session_free = [](QDMI_Device_Session session) {
      asSession(session)->owner->closeSession(session);
    };
    api->device_session_set_parameter =
        [](QDMI_Device_Session session,
           const QDMI_Device_Session_Parameter parameter, const size_t size,
           const void* value) -> int {
      if (parameter != QDMI_DEVICE_SESSION_PARAMETER_CHILDDEVICE) {
        return QDMI_SUCCESS;
      }
      if (value == nullptr || size != sizeof(QDMI_Child_Device)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      auto* typed = asSession(session);
      if (typed->owner->behavior == ChildBehavior::SelectionFailure) {
        return QDMI_ERROR_BADSTATE;
      }
      std::memcpy(&typed->child, value, sizeof(typed->child));
      return QDMI_SUCCESS;
    };
    api->device_session_create_device_job = [](QDMI_Device_Session session,
                                               QDMI_Device_Job* job) -> int {
      auto* owner = asSession(session)->owner;
      owner->job_.owner = owner;
      *job = reinterpret_cast<QDMI_Device_Job>(&owner->job_);
      return QDMI_SUCCESS;
    };
    api->device_job_free = [](QDMI_Device_Job) {};
    api->device_job_set_parameter = [](QDMI_Device_Job job,
                                       QDMI_Device_Job_Parameter parameter,
                                       size_t size, const void* value) {
      return asJob(job)->owner->setJobParameter(job, parameter, size, value);
    };
    api->device_job_query_property =
        [](QDMI_Device_Job job, QDMI_Device_Job_Property property, size_t size,
           void* value, size_t* sizeRet) {
          return asJob(job)->owner->queryJobProperty(job, property, size, value,
                                                     sizeRet);
        };
    api->device_job_submit = [](QDMI_Device_Job job) -> int {
      asJob(job)->owner->submitJob(job);
      return QDMI_SUCCESS;
    };
    api->device_job_cancel = [](QDMI_Device_Job job) -> int {
      asJob(job)->owner->cancelJob(job);
      return QDMI_SUCCESS;
    };
    api->device_job_check = [](QDMI_Device_Job job,
                               QDMI_Job_Status* status) -> int {
      *status = asJob(job)->owner->checkJob(job);
      return QDMI_SUCCESS;
    };
    api->device_job_wait = [](QDMI_Device_Job job, size_t timeout) -> int {
      return asJob(job)->owner->waitJob(job, timeout) ? QDMI_SUCCESS
                                                      : QDMI_ERROR_TIMEOUT;
    };
    api->device_job_get_results = [](QDMI_Device_Job job,
                                     QDMI_Job_Result result, size_t size,
                                     void* data, size_t* sizeRet) {
      return asJob(job)->owner->getJobResult(job, result, size, data, sizeRet);
    };
    api->device_session_query_device_property =
        [](QDMI_Device_Session session, QDMI_Device_Property property,
           size_t size, void* value, size_t* sizeRet) {
          return asSession(session)->owner->queryDevice(session, property, size,
                                                        value, sizeRet);
        };
    api->device_session_query_site_property =
        [](QDMI_Device_Session session, QDMI_Site site,
           QDMI_Site_Property property, size_t size, void* value,
           size_t* sizeRet) {
          return asSession(session)->owner->querySite(session, site, property,
                                                      size, value, sizeRet);
        };
    api->device_session_query_operation_property =
        [](QDMI_Device_Session session, QDMI_Operation operation,
           size_t numSites, const QDMI_Site* sites, size_t numParams,
           const double* params, QDMI_Operation_Property property, size_t size,
           void* value, size_t* sizeRet) {
          return asSession(session)->owner->queryOperation(
              session, operation, numSites, sites, numParams, params, property,
              size, value, sizeRet);
        };
    return api;
  }

  void closeSession(QDMI_Device_Session session) const noexcept {
    if (session == nullptr) {
      return;
    }
    auto* typed = asSession(session);
    if (typed->child == nullptr) {
      closeOrder.emplace_back("parent");
    } else {
      const auto* child = reinterpret_cast<const Child*>(typed->child);
      closeOrder.emplace_back("child-" + std::to_string(child->id));
    }
    ++closed;
    delete typed;
  }

  [[nodiscard]] auto setJobParameter(QDMI_Device_Job, QDMI_Device_Job_Parameter,
                                     size_t, const void*) const -> int {
    return QDMI_SUCCESS;
  }
  [[nodiscard]] auto queryJobProperty(QDMI_Device_Job,
                                      const QDMI_Device_Job_Property property,
                                      const size_t size, void* value,
                                      size_t* sizeRet) const -> int {
    if (property != QDMI_DEVICE_JOB_PROPERTY_PROGRAMFORMAT) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    if (sizeRet != nullptr) {
      *sizeRet = sizeof(programFormat);
    }
    if (value != nullptr) {
      if (size < sizeof(programFormat)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      std::memcpy(value, &programFormat, sizeof(programFormat));
    }
    return QDMI_SUCCESS;
  }
  void submitJob(QDMI_Device_Job) const {}
  void cancelJob(QDMI_Device_Job) const {}
  [[nodiscard]] auto checkJob(QDMI_Device_Job) const -> QDMI_Job_Status {
    return jobStatus;
  }
  [[nodiscard]] auto waitJob(QDMI_Device_Job, size_t) const -> bool {
    return false;
  }
  [[nodiscard]] auto getJobResult(QDMI_Device_Job, QDMI_Job_Result, size_t,
                                  void*, size_t*) const -> int {
    return QDMI_ERROR_NOTSUPPORTED;
  }

  [[nodiscard]] auto queryDevice(QDMI_Device_Session session,
                                 QDMI_Device_Property property,
                                 const size_t size, void* value,
                                 size_t* sizeRet) const -> int {
    const auto* typed = asSession(session);
    if (property == QDMI_DEVICE_PROPERTY_CHILDDEVICES) {
      if (typed->child != nullptr || behavior == ChildBehavior::Unsupported) {
        return QDMI_ERROR_NOTSUPPORTED;
      }
      if (behavior == ChildBehavior::QueryFailure) {
        return QDMI_ERROR_BADSTATE;
      }
      const auto required = behavior == ChildBehavior::Malformed
                                ? sizeof(QDMI_Child_Device) + 1
                                : children_.size() * sizeof(QDMI_Child_Device);
      if (sizeRet != nullptr) {
        *sizeRet = required;
      }
      if (value != nullptr) {
        if (size < required || behavior == ChildBehavior::Malformed) {
          return QDMI_ERROR_INVALIDARGUMENT;
        }
        auto* firstChild = children_.data();
        // The fixed-size fake owns two contiguous child records.
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        auto* secondChild = children_.data() + 1;
        std::array<QDMI_Child_Device, 2> handles{
            reinterpret_cast<QDMI_Child_Device>(firstChild),
            reinterpret_cast<QDMI_Child_Device>(secondChild)};
        std::memcpy(value, static_cast<const void*>(handles.data()), required);
      }
      return QDMI_SUCCESS;
    }
    if (property == QDMI_DEVICE_PROPERTY_STATUS) {
      if (sizeRet != nullptr) {
        *sizeRet = sizeof(deviceStatus);
      }
      if (value != nullptr) {
        if (size < sizeof(deviceStatus)) {
          return QDMI_ERROR_INVALIDARGUMENT;
        }
        std::memcpy(value, &deviceStatus, sizeof(deviceStatus));
      }
      return QDMI_SUCCESS;
    }
    if (property == QDMI_DEVICE_PROPERTY_SUPPORTEDPROGRAMFORMATS) {
      if (sizeRet != nullptr) {
        *sizeRet = sizeof(programFormat);
      }
      if (value != nullptr) {
        if (size < sizeof(programFormat)) {
          return QDMI_ERROR_INVALIDARGUMENT;
        }
        std::memcpy(value, &programFormat, sizeof(programFormat));
      }
      return QDMI_SUCCESS;
    }
    if (property != QDMI_DEVICE_PROPERTY_NAME) {
      return QDMI_ERROR_NOTSUPPORTED;
    }
    const auto name =
        typed->child == nullptr
            ? std::string("parent")
            : "child-" + std::to_string(
                             reinterpret_cast<const Child*>(typed->child)->id);
    if (sizeRet != nullptr) {
      *sizeRet = name.size() + 1;
    }
    if (value != nullptr) {
      if (size < name.size() + 1) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      std::memcpy(value, name.c_str(), name.size() + 1);
    }
    return QDMI_SUCCESS;
  }

  [[nodiscard]] auto querySite(QDMI_Device_Session, QDMI_Site,
                               QDMI_Site_Property, size_t, void*, size_t*) const
      -> int {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  [[nodiscard]] auto queryOperation(QDMI_Device_Session, QDMI_Operation, size_t,
                                    const QDMI_Site*, size_t, const double*,
                                    QDMI_Operation_Property, size_t, void*,
                                    size_t*) const -> int {
    return QDMI_ERROR_NOTSUPPORTED;
  }
};
thread_local const ScriptedDeviceApi* ScriptedDeviceApi::activeApi = nullptr;
// NOLINTEND(readability-named-parameter,cppcoreguidelines-owning-memory)

TEST(DeviceApiTest, ChildRetainsParentSessionAndLibrary) {
  const auto api = std::make_shared<ScriptedDeviceApi>();
  std::optional<Device> retainedChild;
  {
    const auto parent = DeviceFactory::create(api->deviceApi());
    EXPECT_EQ(parent.getName(), "parent");
    const auto children = parent.getChildDevices();
    ASSERT_EQ(children.size(), 2);
    EXPECT_EQ(children[0].getName(), "child-0");
    EXPECT_EQ(children[1].getName(), "child-1");
    retainedChild = children[0];
  }
  ASSERT_EQ(api->closeOrder.size(), 1);
  EXPECT_EQ(api->closeOrder.front(), "child-1");
  retainedChild.reset();
  ASSERT_EQ(api->closeOrder.size(), 3);
  EXPECT_EQ(api->closeOrder.back(), "parent");
  EXPECT_EQ(api->opened, api->closed);
}

TEST(DeviceApiTest, ReusesCanonicalLibrariesProcessWide) {
  const std::filesystem::path library = SC_DEVICE_LIBRARY;
  const auto first = loadDeviceApi(library, "MQT_SC");
  const auto alias = library.parent_path() / "." / library.filename();
  const auto second = loadDeviceApi(alias, "MQT_SC");
  EXPECT_EQ(first, second);
}

TEST(DeviceApiTest, HandlesUnsupportedAndInvalidChildLists) {
  const auto unsupported = std::make_shared<ScriptedDeviceApi>();
  unsupported->behavior = ScriptedDeviceApi::ChildBehavior::Unsupported;
  EXPECT_TRUE(DeviceFactory::create(unsupported->deviceApi())
                  .getChildDevices()
                  .empty());
  EXPECT_EQ(unsupported->opened, unsupported->closed);

  const auto malformed = std::make_shared<ScriptedDeviceApi>();
  malformed->behavior = ScriptedDeviceApi::ChildBehavior::Malformed;
  EXPECT_THROW(static_cast<void>(DeviceFactory::create(malformed->deviceApi())),
               std::runtime_error);
  EXPECT_EQ(malformed->opened, malformed->closed);

  const auto failed = std::make_shared<ScriptedDeviceApi>();
  failed->behavior = ScriptedDeviceApi::ChildBehavior::QueryFailure;
  EXPECT_THROW(static_cast<void>(DeviceFactory::create(failed->deviceApi())),
               std::runtime_error);
  EXPECT_EQ(failed->opened, failed->closed);
}

TEST(DeviceApiTest, CleansUpWhenChildSelectionFails) {
  const auto api = std::make_shared<ScriptedDeviceApi>();
  api->behavior = ScriptedDeviceApi::ChildBehavior::SelectionFailure;
  EXPECT_THROW(static_cast<void>(DeviceFactory::create(api->deviceApi())),
               std::runtime_error);
  EXPECT_EQ(api->opened, api->closed);
}

TEST(DeviceApiTest, RejectsInvalidCustomPropertySelector) {
  const auto api = std::make_shared<ScriptedDeviceApi>();
  api->behavior = ScriptedDeviceApi::ChildBehavior::Unsupported;
  const auto device = DeviceFactory::create(api->deviceApi());
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  constexpr auto invalid = static_cast<CustomProperty>(0);
  EXPECT_THROW(static_cast<void>(device.queryCustomProperty<int>(invalid)),
               std::invalid_argument);
}

TEST(DeviceApiTest, ReturnsQdmiEnumValuesWithoutRedefiningThem) {
  const auto api = std::make_shared<ScriptedDeviceApi>();
  api->behavior = ScriptedDeviceApi::ChildBehavior::Unsupported;
  const auto device = DeviceFactory::create(api->deviceApi());
  // QDMI owns these enum types and values; MQT forwards them unchanged.
  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
  api->deviceStatus = static_cast<QDMI_Device_Status>(QDMI_DEVICE_STATUS_MAX);
  EXPECT_EQ(device.getStatus(), api->deviceStatus);
  api->programFormat =
      static_cast<QDMI_Program_Format>(QDMI_PROGRAM_FORMAT_MAX);
  EXPECT_EQ(device.getSupportedProgramFormats(),
            std::vector{api->programFormat});
  auto job = device.submitJob("", QDMI_PROGRAM_FORMAT_QASM3, 1);
  api->jobStatus = static_cast<QDMI_Job_Status>(1234);
  EXPECT_EQ(job.check(), api->jobStatus);
  EXPECT_EQ(job.getProgramFormat(), api->programFormat);
  constexpr auto providerDefinedFormat = static_cast<QDMI_Program_Format>(1234);
  EXPECT_NO_THROW(
      static_cast<void>(device.submitJob("", providerDefinedFormat, 1)));
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
}
} // namespace
} // namespace qdmi::detail
