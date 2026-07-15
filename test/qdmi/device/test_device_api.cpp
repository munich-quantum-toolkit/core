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
#include "qdmi/Device.hpp"
#include "qdmi/DeviceManager.hpp"

#include <gtest/gtest.h>
#include <qdmi/device.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace qdmi::detail {
namespace {
static_assert(
    std::is_same_v<decltype(DeviceApi::Functions::setJobParameter),
                   ContextFunctionT<decltype(QDMI_device_job_set_parameter)>>);
static_assert(std::is_same_v<
              decltype(DeviceApi::Functions::queryOperation),
              ContextFunctionT<
                  decltype(QDMI_device_session_query_operation_property)>>);

// The fake implements a C ABI whose opaque handles intentionally require
// ownership and pointer casts at this isolated test boundary.
// NOLINTBEGIN(readability-named-parameter,cppcoreguidelines-owning-memory)
class ScriptedDeviceApi final
    : public std::enable_shared_from_this<ScriptedDeviceApi> {
  struct Child {
    size_t id;
  };
  struct Session {
    QDMI_Child_Device child = nullptr;
  };
  struct ScriptedJob {};

  mutable std::array<Child, 2> children_{{{0}, {1}}};
  mutable ScriptedJob job_;

  [[nodiscard]] static auto asSession(QDMI_Device_Session session) -> Session* {
    return reinterpret_cast<Session*>(session);
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
    const auto context = shared_from_this();
    return std::make_shared<const DeviceApi>(DeviceApi{
        .context = context,
        .functions = {
            .openSession =
                [](const void* ctx, const SessionParameters& parameters,
                   const QDMI_Child_Device child) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)
                      ->openSession(parameters, child);
                },
            .closeSession =
                [](const void* ctx,
                   const QDMI_Device_Session session) noexcept {
                  static_cast<const ScriptedDeviceApi*>(ctx)->closeSession(
                      session);
                },
            .createJob =
                [](const void* ctx, const QDMI_Device_Session session) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)->createJob(
                      session);
                },
            .freeJob =
                [](const void* ctx, const QDMI_Device_Job job) noexcept {
                  static_cast<const ScriptedDeviceApi*>(ctx)->freeJob(job);
                },
            .setJobParameter =
                [](const void* ctx, const QDMI_Device_Job job,
                   const QDMI_Device_Job_Parameter parameter, const size_t size,
                   const void* value) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)
                      ->setJobParameter(job, parameter, size, value);
                },
            .queryJobProperty =
                [](const void* ctx, const QDMI_Device_Job job,
                   const QDMI_Device_Job_Property property, const size_t size,
                   void* value, size_t* sizeRet) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)
                      ->queryJobProperty(job, property, size, value, sizeRet);
                },
            .submitJob =
                [](const void* ctx, const QDMI_Device_Job job) {
                  static_cast<const ScriptedDeviceApi*>(ctx)->submitJob(job);
                },
            .cancelJob =
                [](const void* ctx, const QDMI_Device_Job job) {
                  static_cast<const ScriptedDeviceApi*>(ctx)->cancelJob(job);
                },
            .checkJob =
                [](const void* ctx, const QDMI_Device_Job job) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)->checkJob(
                      job);
                },
            .waitJob =
                [](const void* ctx, const QDMI_Device_Job job,
                   const size_t timeout) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)->waitJob(
                      job, timeout);
                },
            .getJobResult =
                [](const void* ctx, const QDMI_Device_Job job,
                   const QDMI_Job_Result result, const size_t size, void* data,
                   size_t* sizeRet) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)
                      ->getJobResult(job, result, size, data, sizeRet);
                },
            .queryDevice =
                [](const void* ctx, const QDMI_Device_Session session,
                   const QDMI_Device_Property property, const size_t size,
                   void* value, size_t* sizeRet) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)
                      ->queryDevice(session, property, size, value, sizeRet);
                },
            .querySite =
                [](const void* ctx, const QDMI_Device_Session session,
                   const QDMI_Site site, const QDMI_Site_Property property,
                   const size_t size, void* value, size_t* sizeRet) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)->querySite(
                      session, site, property, size, value, sizeRet);
                },
            .queryOperation =
                [](const void* ctx, const QDMI_Device_Session session,
                   const QDMI_Operation operation, const size_t numSites,
                   const QDMI_Site* sites, const size_t numParams,
                   const double* params, const QDMI_Operation_Property property,
                   const size_t size, void* value, size_t* sizeRet) {
                  return static_cast<const ScriptedDeviceApi*>(ctx)
                      ->queryOperation(session, operation, numSites, sites,
                                       numParams, params, property, size, value,
                                       sizeRet);
                },
        }});
  }

  [[nodiscard]] auto openSession(const SessionParameters& parameters,
                                 QDMI_Child_Device child) const
      -> QDMI_Device_Session {
    static_cast<void>(parameters);
    ++opened;
    if (child != nullptr && behavior == ChildBehavior::SelectionFailure) {
      ++closed;
      throw std::runtime_error("child selection failed");
    }
    return reinterpret_cast<QDMI_Device_Session>(new Session{child});
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

  [[nodiscard]] auto createJob(QDMI_Device_Session) const -> QDMI_Device_Job {
    return reinterpret_cast<QDMI_Device_Job>(&job_);
  }
  void freeJob(QDMI_Device_Job) const noexcept {}
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

TEST(DeviceApiTest, RejectsUnknownV1EnumValues) {
  const auto api = std::make_shared<ScriptedDeviceApi>();
  api->behavior = ScriptedDeviceApi::ChildBehavior::Unsupported;
  const auto device = DeviceFactory::create(api->deviceApi());
  // These deliberately simulate malformed values crossing the QDMI v1.3 ABI.
  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
  api->deviceStatus = static_cast<QDMI_Device_Status>(QDMI_DEVICE_STATUS_MAX);
  EXPECT_THROW(static_cast<void>(device.getStatus()), std::runtime_error);
  api->programFormat =
      static_cast<QDMI_Program_Format>(QDMI_PROGRAM_FORMAT_MAX);
  EXPECT_THROW(static_cast<void>(device.getSupportedProgramFormats()),
               std::runtime_error);
  auto job = device.submitJob("", ProgramFormat::Qasm3, 1);
  api->jobStatus = static_cast<QDMI_Job_Status>(1234);
  EXPECT_THROW(static_cast<void>(job.check()), std::runtime_error);
  EXPECT_THROW(static_cast<void>(job.getProgramFormat()), std::runtime_error);
  constexpr auto invalidFormat = static_cast<ProgramFormat>(1234);
  EXPECT_THROW(static_cast<void>(device.submitJob("", invalidFormat, 1)),
               std::invalid_argument);
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)
}
} // namespace
} // namespace qdmi::detail
