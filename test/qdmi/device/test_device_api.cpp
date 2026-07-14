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

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qdmi::detail {
namespace {
class ScriptedDeviceApi final : public DeviceApi {
  struct Child {
    size_t id;
  };
  struct Session {
    QDMI_Child_Device child = nullptr;
  };

  std::array<Child, 2> children_{{{0}, {1}}};

  [[nodiscard]] static auto asSession(const QDMI_Device_Session session)
      -> Session* {
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

  ChildBehavior behavior = ChildBehavior::Supported;
  size_t opened = 0;
  size_t closed = 0;
  std::vector<std::string> closeOrder;

  [[nodiscard]] auto openSession(const SessionParameters&,
                                 const QDMI_Child_Device child) const
      -> QDMI_Device_Session override {
    auto* mutableThis = const_cast<ScriptedDeviceApi*>(this);
    ++mutableThis->opened;
    if (child != nullptr && behavior == ChildBehavior::SelectionFailure) {
      ++mutableThis->closed;
      throw std::runtime_error("child selection failed");
    }
    return reinterpret_cast<QDMI_Device_Session>(new Session{child});
  }

  void closeSession(const QDMI_Device_Session session) const noexcept override {
    if (session == nullptr) {
      return;
    }
    auto* mutableThis = const_cast<ScriptedDeviceApi*>(this);
    const auto* typed = asSession(session);
    if (typed->child == nullptr) {
      mutableThis->closeOrder.emplace_back("parent");
    } else {
      const auto* child = reinterpret_cast<const Child*>(typed->child);
      mutableThis->closeOrder.emplace_back("child-" +
                                           std::to_string(child->id));
    }
    ++mutableThis->closed;
    delete typed;
  }

  [[nodiscard]] auto createJob(QDMI_Device_Session) const
      -> QDMI_Device_Job override {
    throw std::runtime_error("jobs are not scripted");
  }
  void freeJob(QDMI_Device_Job) const noexcept override {}
  [[nodiscard]] auto setJobParameter(QDMI_Device_Job, QDMI_Device_Job_Parameter,
                                     size_t, const void*) const
      -> int override {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  [[nodiscard]] auto queryJobProperty(QDMI_Device_Job, QDMI_Device_Job_Property,
                                      size_t, void*, size_t*) const
      -> int override {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  void submitJob(QDMI_Device_Job) const override {
    throw std::runtime_error("jobs are not scripted");
  }
  void cancelJob(QDMI_Device_Job) const override {
    throw std::runtime_error("jobs are not scripted");
  }
  [[nodiscard]] auto checkJob(QDMI_Device_Job) const
      -> QDMI_Job_Status override {
    return QDMI_JOB_STATUS_FAILED;
  }
  [[nodiscard]] auto waitJob(QDMI_Device_Job, size_t) const -> bool override {
    return false;
  }
  [[nodiscard]] auto getJobResult(QDMI_Device_Job, QDMI_Job_Result, size_t,
                                  void*, size_t*) const -> int override {
    return QDMI_ERROR_NOTSUPPORTED;
  }

  [[nodiscard]] auto queryDevice(const QDMI_Device_Session session,
                                 const QDMI_Device_Property property,
                                 const size_t size, void* value,
                                 size_t* sizeRet) const -> int override {
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
        std::array<QDMI_Child_Device, 2> handles{
            reinterpret_cast<QDMI_Child_Device>(
                const_cast<Child*>(&children_[0])),
            reinterpret_cast<QDMI_Child_Device>(
                const_cast<Child*>(&children_[1]))};
        std::memcpy(value, handles.data(), required);
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
      -> int override {
    return QDMI_ERROR_NOTSUPPORTED;
  }
  [[nodiscard]] auto queryOperation(QDMI_Device_Session, QDMI_Operation, size_t,
                                    const QDMI_Site*, size_t, const double*,
                                    QDMI_Operation_Property, size_t, void*,
                                    size_t*) const -> int override {
    return QDMI_ERROR_NOTSUPPORTED;
  }
};

TEST(DeviceApiTest, ChildRetainsParentSessionAndLibrary) {
  const auto api = std::make_shared<ScriptedDeviceApi>();
  std::optional<Device> retainedChild;
  {
    const auto parent = DeviceFactory::create(api);
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
  EXPECT_TRUE(DeviceFactory::create(unsupported).getChildDevices().empty());
  EXPECT_EQ(unsupported->opened, unsupported->closed);

  const auto malformed = std::make_shared<ScriptedDeviceApi>();
  malformed->behavior = ScriptedDeviceApi::ChildBehavior::Malformed;
  EXPECT_THROW(static_cast<void>(DeviceFactory::create(malformed)),
               std::runtime_error);
  EXPECT_EQ(malformed->opened, malformed->closed);

  const auto failed = std::make_shared<ScriptedDeviceApi>();
  failed->behavior = ScriptedDeviceApi::ChildBehavior::QueryFailure;
  EXPECT_THROW(static_cast<void>(DeviceFactory::create(failed)),
               std::runtime_error);
  EXPECT_EQ(failed->opened, failed->closed);
}

TEST(DeviceApiTest, CleansUpWhenChildSelectionFails) {
  const auto api = std::make_shared<ScriptedDeviceApi>();
  api->behavior = ScriptedDeviceApi::ChildBehavior::SelectionFailure;
  EXPECT_THROW(static_cast<void>(DeviceFactory::create(api)),
               std::runtime_error);
  EXPECT_EQ(api->opened, api->closed);
}

TEST(DeviceApiTest, RejectsInvalidCustomPropertySelector) {
  const auto api = std::make_shared<ScriptedDeviceApi>();
  api->behavior = ScriptedDeviceApi::ChildBehavior::Unsupported;
  const auto device = DeviceFactory::create(api);
  constexpr auto invalid = static_cast<CustomProperty>(0);
  EXPECT_THROW(static_cast<void>(device.queryCustomProperty<int>(invalid)),
               std::invalid_argument);
}
} // namespace
} // namespace qdmi::detail
