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

#include <gtest/gtest.h>

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

} // namespace
