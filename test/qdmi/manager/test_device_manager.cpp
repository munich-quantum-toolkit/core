/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/Device.hpp"
#include "qdmi/DeviceManager.hpp"
#include "qdmi/DeviceRegistry.hpp"
#include "qdmi/SessionConfig.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <future>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

qdmi::DeviceDefinition scDefinition(std::string id = "mqt.sc.test") {
  return {
      .id = std::move(id),
      .library = SC_DEVICE_LIBRARY,
      .prefix = "MQT_SC",
  };
}

qdmi::DeviceManager scManager(std::string id = "mqt.sc.test") {
  return qdmi::DeviceManager(
      qdmi::DeviceRegistry({scDefinition(std::move(id))}));
}

qdmi::DeviceDefinition sessionDefinition(std::string id,
                                         std::string custom1 = {}) {
  qdmi::DeviceDefinition definition{
      .id = std::move(id),
      .library = SESSION_DEVICE_LIBRARY,
      .prefix = "TEST_SESSION",
  };
  if (!custom1.empty()) {
    definition.session.custom1 = std::move(custom1);
  }
  return definition;
}

TEST(DeviceRegistry, RegistersReplacesAndProtectsExistingDefinitions) {
  qdmi::DeviceRegistry registry({scDefinition("example")});
  EXPECT_THROW(registry.registerDevice(scDefinition("example")),
               std::invalid_argument);
  EXPECT_FALSE(registry.registerDeviceIfAbsent(scDefinition("example")));

  auto replacement = scDefinition("example");
  replacement.prefix = "REPLACEMENT";
  registry.registerDevice(std::move(replacement), true);
  ASSERT_EQ(registry.definitions().size(), 1);
  EXPECT_EQ(registry.definitions().front().prefix, "REPLACEMENT");

  EXPECT_TRUE(registry.registerDeviceIfAbsent(scDefinition("fallback")));
  ASSERT_EQ(registry.definitions().size(), 2);
  EXPECT_EQ(registry.definitions().back().id, "fallback");
}

TEST(DeviceRegistry, RejectsIncompleteDefinitions) {
  qdmi::DeviceRegistry registry(std::vector<qdmi::DeviceDefinition>{});
  EXPECT_THROW(registry.registerDevice({}), std::invalid_argument);
  EXPECT_THROW(
      registry.registerDevice({.id = "missing-library", .prefix = "MQT_SC"}),
      std::invalid_argument);
  EXPECT_THROW(registry.registerDevice(
                   {.id = "missing-prefix", .library = SC_DEVICE_LIBRARY}),
               std::invalid_argument);
}

TEST(DeviceManager, LazilyOpensAndKeepsDeviceAlive) {
  const auto device = scManager().open("mqt.sc.test");
  EXPECT_EQ(device.getName(), "MQT SC Default QDMI Device");
  EXPECT_FALSE(device.getSites().empty());
}

TEST(DeviceManager, OpensDefinitionsIndividually) {
  qdmi::DeviceRegistry registry({
      scDefinition("good"),
      {.id = "bad", .library = "does-not-exist", .prefix = "MISSING"},
  });
  const qdmi::DeviceManager manager(std::move(registry));

  EXPECT_EQ(manager.open("good").getName(), "MQT SC Default QDMI Device");
  EXPECT_THROW(static_cast<void>(manager.open("bad")), std::runtime_error);
  EXPECT_THROW(static_cast<void>(manager.open("missing")), std::out_of_range);
}

TEST(DeviceManager, OpensAllDefinitionsAndIsolatesFailures) {
  qdmi::DeviceRegistry registry({
      scDefinition("good"),
      {.id = "bad", .library = "does-not-exist", .prefix = "MISSING"},
  });
  const qdmi::DeviceManager manager(std::move(registry));
  const auto result = manager.openAll();

  ASSERT_EQ(result.devices.size(), 1);
  EXPECT_EQ(result.devices.at("good").getName(), "MQT SC Default QDMI Device");
  ASSERT_EQ(result.errors.size(), 1);
  EXPECT_FALSE(result.errors.at("bad").empty());
}

TEST(DeviceManager, ConcurrentOpenCallsShareTheLibrarySafely) {
  const auto manager = scManager("concurrent");
  std::vector<std::future<std::string>> names;
  names.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    names.emplace_back(std::async(std::launch::async, [&manager] {
      return manager.open("concurrent").getName();
    }));
  }
  for (auto& name : names) {
    EXPECT_EQ(name.get(), "MQT SC Default QDMI Device");
  }
}

TEST(DeviceManager, SharesLibrariesButCreatesFreshSessions) {
  qdmi::DeviceRegistry registry(
      {scDefinition("first"), scDefinition("second")});
  const qdmi::DeviceManager manager(std::move(registry));
  const auto first = manager.open("first");
  const auto second = manager.open("second");

  EXPECT_NE(first, second);
  EXPECT_EQ(first.getName(), second.getName());
  EXPECT_EQ(first.getSites().size(), second.getSites().size());
}

TEST(DeviceManager, OpenedObjectsOutliveManager) {
  const qdmi::Device device = [] {
    const auto manager = scManager("persistent");
    return manager.open("persistent");
  }();
  const qdmi::Site site = [&device] { return device.getSites().front(); }();

  EXPECT_EQ(device.getName(), "MQT SC Default QDMI Device");
  EXPECT_EQ(site.getIndex(), 0);
}

TEST(DeviceManager, RejectsIncompleteSymbolSet) {
  auto definition = scDefinition("wrong-prefix");
  definition.prefix = "MISSING";
  const qdmi::DeviceManager manager(
      qdmi::DeviceRegistry({std::move(definition)}));
  EXPECT_THROW(static_cast<void>(manager.open("wrong-prefix")),
               std::runtime_error);
}

TEST(DeviceManager, KeepsAnImmutableRegistrySnapshot) {
  qdmi::DeviceRegistry registry({sessionDefinition("snapshot.first")});
  const qdmi::DeviceManager manager(registry);
  registry.registerDevice(sessionDefinition("snapshot.second"));

  EXPECT_EQ(manager.deviceIds(), std::vector<std::string>{"snapshot.first"});
  EXPECT_THROW(static_cast<void>(manager.open("snapshot.second")),
               std::out_of_range);
}

TEST(DeviceManager, DefaultManagersSnapshotRuntimeRegistrations) {
  const qdmi::DeviceManager before;
  ASSERT_TRUE(qdmi::registerDeviceIfAbsent(
      sessionDefinition("test.default.snapshot", "registered")));
  const qdmi::DeviceManager after;

  EXPECT_THROW(static_cast<void>(before.open("test.default.snapshot")),
               std::out_of_range);
  EXPECT_NE(
      after.open("test.default.snapshot").getName().find("custom1=registered"),
      std::string::npos);
}

TEST(DeviceManager, ReplacementOnlyChangesFutureDefaultOpens) {
  qdmi::registerDevice(sessionDefinition("test.default.replace", "old"), true);
  const auto oldDevice = qdmi::openDevice("test.default.replace");
  qdmi::registerDevice(sessionDefinition("test.default.replace", "new"), true);
  const auto newDevice = qdmi::openDevice("test.default.replace");

  EXPECT_NE(oldDevice.getName().find("custom1=old"), std::string::npos);
  EXPECT_NE(newDevice.getName().find("custom1=new"), std::string::npos);
}

TEST(DeviceManager, PreservesTypedConfigurationAndPerOpenOverrides) {
  auto definition = sessionDefinition("typed.configuration");
  definition.session.deviceConfiguration =
      qdmi::InlineDeviceConfiguration{.json = R"({"model":"default"})"};
  const qdmi::DeviceManager manager(
      qdmi::DeviceRegistry({std::move(definition)}));

  const auto configured = manager.open("typed.configuration");
  EXPECT_NE(configured.getName().find(R"(custom1={"model":"default"})"),
            std::string::npos);

  qdmi::DeviceSessionConfig override;
  override.deviceConfiguration =
      qdmi::FileDeviceConfiguration{.path = "override.json"};
  const auto overridden = manager.open("typed.configuration", override);
  EXPECT_NE(overridden.getName().find("custom2=override.json"),
            std::string::npos);

  override.custom1 = "raw";
  EXPECT_THROW(static_cast<void>(manager.open("typed.configuration", override)),
               std::invalid_argument);
}

TEST(DeviceManager, PreservesCurrentJobAndCustomOperationFeatures) {
  const qdmi::DeviceManager manager(
      qdmi::DeviceRegistry({sessionDefinition("current.features")}));
  const auto device = manager.open("current.features");

  const auto operations =
      device.queryCustomOperations(qdmi::CustomProperty::Custom1);
  ASSERT_TRUE(operations.has_value());
  ASSERT_EQ(operations->size(), 2);
  EXPECT_EQ(operations->front().getName(), "custom-rx");
  EXPECT_EQ(operations->back().getQubitsNum(), 2);
  const auto emptyOperations =
      device.queryCustomOperations(qdmi::CustomProperty::Custom2);
  ASSERT_TRUE(emptyOperations.has_value());
  EXPECT_TRUE(emptyOperations->empty());
  EXPECT_THROW(static_cast<void>(
                   device.queryCustomOperations(qdmi::CustomProperty::Custom3)),
               std::runtime_error);
  EXPECT_FALSE(device.getQueueLength().has_value());

  const auto job = device.retrieveJobById("session-job");
  EXPECT_EQ(job.getId(), "session-job");
  EXPECT_FALSE(job.getQueuePosition().has_value());
  EXPECT_THROW(static_cast<void>(device.retrieveJobById("missing")),
               std::runtime_error);
}

} // namespace
