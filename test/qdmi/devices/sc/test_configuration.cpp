/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/devices/sc/Configuration.hpp"

#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ios>
#include <istream>
#include <limits>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

namespace sc {
namespace {
using Json = llvm::json::Value;

[[nodiscard]] Json bundledJson() {
  return llvm::cantFail(llvm::json::parse(
      (*llvm::MemoryBuffer::getFile(SC_DEVICE_JSON))->getBuffer()));
}

void expectInvalid(const std::function<void(Json&)>& mutate,
                   const std::string& diagnostic) {
  auto value = bundledJson();
  mutate(value);
  auto result = readJSON(llvm::formatv("{0}", value).str(), "test-source");
  ASSERT_FALSE(static_cast<bool>(result));
  const auto error = llvm::toString(result.takeError());
  EXPECT_THAT(error, testing::HasSubstr("test-source"));
  EXPECT_THAT(error, testing::HasSubstr(diagnostic));
}
} // namespace

TEST(ScConfigurationTest, ParsesBundledDeviceStrictly) {
  const auto device = llvm::cantFail(readJSON(SC_DEVICE_JSON));
  EXPECT_EQ(device.schemaVersion, 1);
  EXPECT_EQ(device.name, "MQT SC Default QDMI Device");
  EXPECT_EQ(device.numQubits, 100);
  EXPECT_EQ(device.operations.size(), 3);
  EXPECT_EQ(device.operations[0].name, "r");
  EXPECT_EQ(device.operations[1].name, "cz");
  EXPECT_EQ(device.operations[2].name, "measure");
  ASSERT_EQ(device.couplings.size(), 180);
  EXPECT_EQ(device.couplings.front(), (std::pair<uint64_t, uint64_t>{0, 1}));
  EXPECT_EQ(device.couplings.back(), (std::pair<uint64_t, uint64_t>{89, 99}));
}

TEST(ScConfigurationTest, ParsesIqmDeviceModels) {
  const auto assertModel = [](const Device& device, const std::string& name,
                              const size_t qubits, const size_t couplings,
                              const size_t czSiteTuples) {
    EXPECT_EQ(device.name, name);
    EXPECT_EQ(device.numQubits, qubits);
    EXPECT_EQ(device.couplings.size(), couplings);
    EXPECT_EQ(device.durationUnit.unit, "us");
    EXPECT_DOUBLE_EQ(device.durationUnit.scaleFactor, 0.001);
    ASSERT_EQ(device.qubitProperties.overrides.size(), qubits);
    for (size_t i = 0; i < qubits; ++i) {
      const auto& qubit = device.qubitProperties.overrides[i];
      EXPECT_EQ(qubit.qubit, i);
      ASSERT_TRUE(qubit.name);
      EXPECT_EQ(*qubit.name, "QB" + std::to_string(i + 1));
    }

    ASSERT_EQ(device.operations.size(), 3);
    EXPECT_EQ(device.operations[0].name, "r");
    EXPECT_EQ(device.operations[0].numParameters, 2);
    EXPECT_EQ(device.operations[0].numQubits, 1);
    EXPECT_EQ(device.operations[0].siteOverrides.size(), qubits);
    EXPECT_EQ(device.operations[1].name, "cz");
    EXPECT_EQ(device.operations[1].numParameters, 0);
    EXPECT_EQ(device.operations[1].numQubits, 2);
    EXPECT_EQ(device.operations[1].siteOverrides.size(), czSiteTuples);
    EXPECT_EQ(device.operations[2].name, "measure");
    EXPECT_EQ(device.operations[2].numParameters, 0);
    EXPECT_EQ(device.operations[2].numQubits, 1);
    EXPECT_EQ(device.operations[2].siteOverrides.size(), qubits);
    for (const auto& operation : device.operations) {
      EXPECT_FALSE(operation.duration);
      for (const auto& siteOverride : operation.siteOverrides) {
        EXPECT_FALSE(siteOverride.duration);
        EXPECT_TRUE(siteOverride.fidelity);
      }
    }
  };

  const auto garnet = llvm::cantFail(readJSON(IQM_GARNET_JSON));
  assertModel(garnet, "IQM Garnet", 20, 30, 30);
  EXPECT_EQ(std::ranges::count_if(
                garnet.qubitProperties.overrides,
                [](const auto& qubit) { return qubit.t1.has_value(); }),
            20);
  EXPECT_EQ(std::ranges::count_if(
                garnet.qubitProperties.overrides,
                [](const auto& qubit) { return qubit.t2.has_value(); }),
            20);

  const auto emerald = llvm::cantFail(readJSON(IQM_EMERALD_JSON));
  assertModel(emerald, "IQM Emerald", 54, 90, 81);
  EXPECT_EQ(std::ranges::count_if(
                emerald.qubitProperties.overrides,
                [](const auto& qubit) { return qubit.t1.has_value(); }),
            53);
  EXPECT_EQ(std::ranges::count_if(
                emerald.qubitProperties.overrides,
                [](const auto& qubit) { return qubit.t2.has_value(); }),
            54);
}

TEST(ScConfigurationTest, RejectsTopLevelSchemaErrors) {
  expectInvalid([](auto& root) { (*root.getAsObject())["unknown"] = true; },
                "unknown");
  expectInvalid(
      [](auto& root) { root.getAsObject()->erase("qubitProperties"); },
      "qubitProperties");
  expectInvalid([](auto& root) { (*root.getAsObject())["schema-version"] = 2; },
                "schema-version");
  expectInvalid([](auto& root) { (*root.getAsObject())["name"] = ""; }, "name");
  expectInvalid([](auto& root) { (*root.getAsObject())["numQubits"] = 0; },
                "numQubits");
  expectInvalid([](auto& root) { (*root.getAsObject())["numQubits"] = "100"; },
                "numQubits");
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject())["numQubits"] =
            std::numeric_limits<uint64_t>::max();
      },
      "numQubits");
}

TEST(ScConfigurationTest, RejectsInvalidUnitsAndCalibration) {
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject()->getObject("durationUnit"))["unit"] = "fortnight";
      },
      "durationUnit");
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject()->getObject("durationUnit"))["scaleFactor"] = 0.;
      },
      "durationUnit");
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject()->getObject("qubitProperties"))["overrides"]
            .getAsArray()
            ->push_back(llvm::json::Object{{"qubit", 100}, {"t1", 1}});
      },
      "qubitProperties/overrides");
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject()->getObject("qubitProperties"))["overrides"]
            .getAsArray()
            ->push_back(llvm::json::Object{{"qubit", 8}, {"t1", 0}});
      },
      "qubitProperties/overrides");
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject()->getObject("qubitProperties"))["overrides"]
            .getAsArray()
            ->push_back(llvm::json::Object{{"qubit", 8}, {"name", ""}});
      },
      "qubitProperties/overrides");
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject()->getObject("qubitProperties"))["overrides"]
            .getAsArray()
            ->push_back(llvm::json::Object{{"qubit", 8}});
      },
      "qubitProperties/overrides");
}

TEST(ScConfigurationTest, AcceptsOptionalSiteNames) {
  auto root = bundledJson();
  (*root.getAsObject()->getObject("qubitProperties"))["overrides"]
      .getAsArray()
      ->push_back(llvm::json::Object{{"qubit", 8}, {"name", "QB9"}});
  const auto device =
      llvm::cantFail(readJSON(llvm::formatv("{0}", root).str(), "site-name"));
  const auto& qubitOverride = device.qubitProperties.overrides.back();
  EXPECT_EQ(qubitOverride.qubit, 8);
  ASSERT_TRUE(qubitOverride.name);
  EXPECT_EQ(*qubitOverride.name, "QB9");
  EXPECT_FALSE(qubitOverride.t1);
  EXPECT_FALSE(qubitOverride.t2);
}

TEST(ScConfigurationTest, RejectsNamesContainingNul) {
  for (const auto& name :
       {std::string("\0hidden", 7), std::string("x\0different", 11)}) {
    expectInvalid([&name](auto& root) { (*root.getAsObject())["name"] = name; },
                  "name");
    expectInvalid(
        [&name](auto& root) {
          (*root.getAsObject()->getObject("qubitProperties"))["overrides"]
              .getAsArray()
              ->push_back(llvm::json::Object{{"qubit", 8}, {"name", name}});
        },
        "qubitProperties/overrides");
    expectInvalid(
        [&name](auto& root) {
          (*(*root.getAsObject()->getArray("operations"))[0]
                .getAsObject())["name"] = name;
        },
        "operations");
  }
}

TEST(ScConfigurationTest, RejectsInvalidOrderedTopology) {
  expectInvalid(
      [](auto& root) {
        root.getAsObject()
            ->getArray("couplings")
            ->push_back((*root.getAsObject()->getArray("couplings"))[0]);
      },
      "couplings");
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject()->getArray("couplings"))[0] =
            llvm::json::Array{0, 0};
      },
      "couplings");
  expectInvalid(
      [](auto& root) {
        (*root.getAsObject()->getArray("couplings"))[0] =
            llvm::json::Array{0, 100};
      },
      "couplings");

  auto root = bundledJson();
  (*root.getAsObject())["couplings"] =
      llvm::json::Array{llvm::json::Array{1, 0}, llvm::json::Array{0, 1}};
  (*(*root.getAsObject()->getArray("operations"))[1]
        .getAsObject())["siteOverrides"] = llvm::json::Array{};
  const auto parsed =
      llvm::cantFail(readJSON(llvm::formatv("{0}", root).str(), "ordered"));
  EXPECT_EQ(parsed.couplings[0], (std::pair<uint64_t, uint64_t>{1, 0}));
  EXPECT_EQ(parsed.couplings[1], (std::pair<uint64_t, uint64_t>{0, 1}));
}

TEST(ScConfigurationTest, ValidatesOperationSupportAndCalibration) {
  expectInvalid(
      [](auto& root) {
        root.getAsObject()
            ->getArray("operations")
            ->push_back(root.getAsObject()->getArray("operations")->front());
      },
      "operations");
  expectInvalid(
      [](auto& root) {
        (*(*root.getAsObject()->getArray("operations"))[0]
              .getAsObject())["numQubits"] = 3;
        (*root.getAsObject()->getArray("operations"))[0].getAsObject()->erase(
            "sites");
      },
      "sites");
  expectInvalid(
      [](auto& root) {
        (*(*root.getAsObject()->getArray("operations"))[1]
              .getAsObject())["sites"] =
            llvm::json::Array{llvm::json::Array{0, 2}};
        (*(*root.getAsObject()->getArray("operations"))[1]
              .getAsObject())["siteOverrides"] = llvm::json::Array{};
      },
      "device connectivity");
  expectInvalid(
      [](auto& root) {
        (*(*root.getAsObject()->getArray("operations"))[0]
              .getAsObject())["fidelity"] = 1.1;
      },
      "fidelity");
  expectInvalid(
      [](auto& root) {
        (*(*(*root.getAsObject()->getArray("operations"))[1]
                .getAsObject()
                ->getArray("siteOverrides"))[0]
              .getAsObject())["sites"] = llvm::json::Array{1, 0};
      },
      "siteOverrides");
  expectInvalid(
      [](auto& root) {
        (*(*root.getAsObject()->getArray("operations"))[1]
              .getAsObject()
              ->getArray("siteOverrides"))[0]
            .getAsObject()
            ->erase("duration");
        (*(*root.getAsObject()->getArray("operations"))[1]
              .getAsObject()
              ->getArray("siteOverrides"))[0]
            .getAsObject()
            ->erase("fidelity");
      },
      "siteOverrides");
}

TEST(ScConfigurationTest, AcceptsZeroDurationAndRejectsOverflowingJsonNumber) {
  auto root = bundledJson();
  (*(*root.getAsObject()->getArray("operations"))[0]
        .getAsObject())["duration"] = 0;
  (*(*(*root.getAsObject()->getArray("operations"))[1].getAsObject()->getArray(
      "siteOverrides"))[0]
        .getAsObject())["duration"] = 0;
  EXPECT_NO_THROW(static_cast<void>(llvm::cantFail(
      readJSON(llvm::formatv("{0}", root).str(), "zero-duration"))));
  EXPECT_TRUE(llvm::errorToBool(
      readJSON(
          R"({"schema-version":1e400,"name":"x","numQubits":1,"durationUnit":{"unit":"ns","scaleFactor":1},"qubitProperties":{"defaults":{},"overrides":[]},"couplings":[],"operations":[]})",
          "overflow")
          .takeError()));
}

TEST(ScConfigurationTest, AcceptsExplicitHigherAritySites) {
  auto root = bundledJson();
  (*root.getAsObject())["numQubits"] = 3;
  (*root.getAsObject())["couplings"] = llvm::json::Array{};
  (*root.getAsObject()->getObject("qubitProperties"))["overrides"] =
      llvm::json::Array{};
  (*root.getAsObject())["operations"] = llvm::json::Array{
      llvm::json::Object{
          {"name", "three"},
          {"numParameters", 1},
          {"numQubits", 3},
          {"sites", llvm::json::Array{llvm::json::Array{2, 0, 1}}},
          {"duration", 0},
          {
              "siteOverrides",
              llvm::json::Array{
                  llvm::json::Object{
                      {"sites", llvm::json::Array{2, 0, 1}},
                      {"fidelity", 0.75},
                  },
              },
          },
      },
  };
  const auto parsed = llvm::cantFail(
      readJSON(llvm::formatv("{0}", root).str(), "higher-arity"));
  ASSERT_EQ(parsed.operations.size(), 1);
  ASSERT_TRUE(parsed.operations[0].sites.has_value());
  EXPECT_EQ(parsed.operations[0].sites->front(),
            (std::vector<uint64_t>{2, 0, 1}));
}

TEST(ScConfigurationTest, ReportsStreamReadFailures) {
  class FailingBuffer final : public std::streambuf {
  protected:
    int_type underflow() override {
      throw std::ios_base::failure("injected read failure");
    }
  } buffer;
  std::istream stream(&buffer);
  auto parsed = readJSON(stream, "failing-stream");
  ASSERT_FALSE(static_cast<bool>(parsed));
  const auto message = llvm::toString(parsed.takeError());
  EXPECT_THAT(message, testing::HasSubstr("failing-stream"));
  EXPECT_THAT(message, testing::HasSubstr("injected read failure"));
}

} // namespace sc
