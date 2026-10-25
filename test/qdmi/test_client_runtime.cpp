/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/QDMI.hpp"
#include "qdmi/common/Common.hpp"
#include "qdmi/common/DeviceConfiguration.hpp"

#include "support/TestSupport.hpp"

/// Required for JSON construction inside GoogleTest macro expansions.

#include "qdmi/constants.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <optional>
#include <initializer_list>
#include <string>
#include <system_error>
#include <vector>

namespace qdmi {
namespace {
using mqt::test::ScopedEnvironmentVariable;

TEST(ClientRuntimeTest, LocatesAbsoluteAndRelativeLoadedLibraries) {
  const mqt::test::TemporaryDirectory directory;
  const std::filesystem::path fixture{MQT_CORE_QDMI_INCOMPLETE_DRIVER};
  // Windows searches the executable directory before the current directory.
  const auto libraryPath =
      directory.path() / ("relocated-" + fixture.filename().string());
  std::filesystem::copy_file(fixture, libraryPath);
  // The temporary directory can be on a different Windows drive.
  const auto previous = std::filesystem::current_path();
  const auto restore = llvm::make_scope_exit([&] {
    std::error_code error;
    std::filesystem::current_path(previous, error);
    EXPECT_FALSE(error) << error.message();
  });
  std::filesystem::current_path(directory.path());
  // Load relative first; the loader may retain the image after dlclose.
  for (const auto& path :
       {std::filesystem::path(".") / libraryPath.filename(), libraryPath}) {
    SCOPED_TRACE(path);
    auto encoded = detail::pathToString(path);
    MQT_ASSERT_SUCCESS(encoded);
    std::string error;
    auto library =
        llvm::sys::DynamicLibrary::getLibrary(encoded->c_str(), &error);
    ASSERT_TRUE(library.isValid()) << error;
    const auto close = llvm::make_scope_exit(
        [&] { llvm::sys::DynamicLibrary::closeLibrary(library); });
    const auto* anchor =
        library.getAddressOfSymbol("QDMI_driver_get_client_abi_version");
    ASSERT_NE(anchor, nullptr);
    auto located = detail::moduleDirectory(anchor);
    MQT_ASSERT_SUCCESS(located);
    // GetModuleFileNameW can preserve the temporary path's short 8.3 spelling.
    EXPECT_TRUE(std::filesystem::equivalent(*located, directory.path()));
  }
}

TEST(ClientRuntimeTest, PreservesQDMIErrorStatus) {
  EXPECT_EQ(toStatus(checkError(QDMI_SUCCESS, "success")), QDMI_SUCCESS);
  EXPECT_EQ(toStatus(checkError(QDMI_WARN_GENERAL, "warning")), QDMI_SUCCESS);
  for (const auto status :
       {QDMI_ERROR_INVALIDARGUMENT, QDMI_ERROR_OUTOFMEM, QDMI_ERROR_NOTFOUND}) {
    EXPECT_EQ(toStatus(checkError(status, "provider failure")), status);
  }
}

TEST(ClientRuntimeTest, RejectsMissingModuleAnchors) {
  EXPECT_EQ(toStatus(detail::moduleDirectory(nullptr).takeError()),
            QDMI_ERROR_INVALIDARGUMENT);
  int anchor = 0;
  EXPECT_EQ(toStatus(detail::moduleDirectory(&anchor).takeError()),
            QDMI_ERROR_FATAL);
}

TEST(ClientRuntimeTest, LoadsUnicodeConfigurationPaths) {
  const mqt::test::TemporaryDirectory directory;
  auto filename = detail::pathFromString("\xc3\xa4-\xf0\x9f\x8e\xaf.json");
  MQT_ASSERT_SUCCESS(filename);
  const auto path = directory.path() / *filename;
  auto encoded = detail::pathToString(path);
  MQT_ASSERT_SUCCESS(encoded);
  std::error_code error;
  {
    llvm::raw_fd_ostream output(*encoded, error);
    ASSERT_FALSE(error) << error.message();
    output << "{}";
  }
  std::optional<std::string> inlineJson;
  std::optional<std::filesystem::path> file;
  EXPECT_EQ(detail::setDeviceConfigurationParameter(
                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2, encoded->size() + 1,
                encoded->c_str(), inlineJson, file),
            QDMI_SUCCESS);
  EXPECT_EQ(file, path);
  auto explicitConfig = detail::loadDeviceConfiguration(
      inlineJson, file, "MQT_CORE_TEST_INLINE_CONFIG",
      "MQT_CORE_TEST_FILE_CONFIG", "unused", nullptr);
  MQT_ASSERT_SUCCESS(explicitConfig);
  EXPECT_EQ(explicitConfig->json, "{}");
  const ScopedEnvironmentVariable noInline{"MQT_CORE_TEST_INLINE_CONFIG",
                                           std::nullopt};
  const ScopedEnvironmentVariable environment{"MQT_CORE_TEST_FILE_CONFIG",
                                              *encoded};
  auto environmentConfig = detail::loadDeviceConfiguration(
      std::nullopt, std::nullopt, "MQT_CORE_TEST_INLINE_CONFIG",
      "MQT_CORE_TEST_FILE_CONFIG", "unused", nullptr);
  MQT_ASSERT_SUCCESS(environmentConfig);
  EXPECT_EQ(environmentConfig->json, "{}");
}

#ifdef _WIN32
TEST(ClientRuntimeTest, RejectsMalformedPathsWithoutReplacingConfiguration) {
  EXPECT_EQ(toStatus(detail::pathFromString("\xff").takeError()),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(toStatus(detail::pathToString(std::filesystem::path(
                                              std::wstring(1, wchar_t{0xd800})))
                         .takeError()),
            QDMI_ERROR_INVALIDARGUMENT);
  std::optional<std::string> inlineJson;
  std::optional<std::filesystem::path> file = "previous.json";
  const std::string malformed = "\xff";
  EXPECT_EQ(detail::setDeviceConfigurationParameter(
                QDMI_DEVICE_SESSION_PARAMETER_CUSTOM2, malformed.size() + 1,
                malformed.c_str(), inlineJson, file),
            QDMI_ERROR_INVALIDARGUMENT);
  EXPECT_EQ(file, std::filesystem::path("previous.json"));
}
#endif

TEST(ClientRuntimeTest, ValidatesDriversAndRetainsSessions) {
  const ScopedEnvironmentVariable config{"QDMI_CONF",
                                         MQT_CORE_QDMI_EXAMPLE_CONFIG};
  const ScopedEnvironmentVariable driver{"MQT_CORE_QDMI_DRIVER",
                                         "missing-driver"};
  EXPECT_THAT(::mqt::test::errorMessage([] { return Session::create(); }),
              testing::HasSubstr("Cannot load QDMI driver"));
  EXPECT_THAT(::mqt::test::errorMessage([] {
                return Session::create(
                    SessionConfig{.driverPath = std::filesystem::path{}});
              }),
              testing::HasSubstr("QDMI driver path must not be empty"));
  EXPECT_THAT(::mqt::test::errorMessage([] {
                return Session::create(SessionConfig{
                    .driverPath = MQT_CORE_QDMI_INCOMPLETE_DRIVER});
              }),
              testing::HasSubstr("missing symbol QDMI_session_alloc"));
  EXPECT_THAT(::mqt::test::errorMessage([] {
                return Session::create(SessionConfig{
                    .driverPath = MQT_CORE_QDMI_INCOMPATIBLE_DRIVER});
              }),
              testing::HasSubstr("incompatible ABI"));

  // The upstream example requires a token. A failed initialization must allow
  // retry.
  EXPECT_THAT(::mqt::test::errorMessage([] {
                return Session::create(
                    SessionConfig{.driverPath = MQT_CORE_QDMI_TEST_DRIVER});
              }),
              testing::HasSubstr("Permission denied"));
  const SessionConfig firstConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "first-token",
  };
  auto first = ::mqt::test::value(Session::create(firstConfig));
  auto second = ::mqt::test::value(Session::create(SessionConfig{
      .driverPath = MQT_CORE_QDMI_TEST_DRIVER,
      .token = "second-token",
  }));
  EXPECT_EQ(::mqt::test::value(first.getDeviceIds()),
            std::vector<std::string>{"test.example"});
  EXPECT_THAT(
      ::mqt::test::value(
          ::mqt::test::value(first.getDevice("test.example")).getName()),
      testing::HasSubstr("token=first-token"));
  EXPECT_THAT(
      ::mqt::test::value(
          ::mqt::test::value(second.getDevice("test.example")).getName()),
      testing::HasSubstr("token=second-token"));

  const auto retained =
      ::mqt::test::value(Session::openDevice("test.example", firstConfig));
  EXPECT_THAT(::mqt::test::value(retained.getName()),
              testing::HasSubstr("token=first-token"));
  const auto job = ::mqt::test::value(
      ::mqt::test::value(Session::openDevice("test.example", firstConfig))
          .submitJob("payload", QDMI_PROGRAM_FORMAT_CUSTOM1));
  EXPECT_EQ(::mqt::test::value(job.getId()), "session-job");
  {
    const ScopedEnvironmentVariable optionalToken{
        "MQT_CORE_QDMI_TEST_DEVICE_FAILURE", "token-unsupported"};
    EXPECT_EQ(
        ::mqt::test::value(
            ::mqt::test::value(Session::openDevice("test.example", firstConfig))
                .getId()),
        "test.example");
  }
  {
    const ScopedEnvironmentVariable deniedToken{
        "MQT_CORE_QDMI_TEST_DEVICE_FAILURE", "token-denied"};
    EXPECT_THAT(
        ::mqt::test::errorMessage([&] { return Session::create(firstConfig); }),
        testing::HasSubstr("Permission denied"));
  }
}

TEST(BuiltinDriverExtensionTest, DiscoversThenOpensIndependentSessions) {
  EXPECT_EQ(
      ::mqt::test::errorStatus([] { return builtin_driver::openDevice(""); }),
      QDMI_ERROR_INVALIDARGUMENT);
  const mqt::test::TemporaryDirectory directory;
  const auto definition = [](const std::string& id) {
    return llvm::json::Object{
        {"id", id},
        {"library", MQT_CORE_QDMI_SESSION_DEVICE},
        {"prefix", "TEST_SESSION"},
        {"session", llvm::json::Object{{"custom4", "idle"}}},
    };
  };
  const auto manifest = [&](const std::filesystem::path& name,
                            const std::string& id) {
    return directory.write(
        name,
        llvm::formatv(
            "{0}", llvm::json::Value(llvm::json::Object{
                       {"schema-version", 1},
                       {
                           "qdmi",
                           llvm::json::Object{
                               {"devices", llvm::json::Array{definition(id)}},
                           },
                       },
                   }))
            .str());
  };
  const auto configured = manifest("device.qdmi.json", "test.session");
  const auto conflicting = manifest("duplicate.qdmi.json", "test.session");
  const auto unicode = manifest(
      std::filesystem::path{u8"device-ünicode.qdmi.json"}, "test.unicode");
  const auto late = manifest("late.qdmi.json", "test.late");
  const auto malformed = directory.write("malformed.qdmi.json", "{");
  auto missingLibrary = definition("test.missing-library");
  missingLibrary["library"] = (directory.path() / "missing-device").string();
  const auto unavailable = directory.write(
      "unavailable.qdmi.json",
      llvm::formatv("{0}",
                    llvm::json::Value(llvm::json::Object{
                        {"schema-version", 1},
                        {
                            "qdmi",
                            llvm::json::Object{
                                {
                                    "devices",
                                    llvm::json::Array{
                                        llvm::json::Object(missingLibrary),
                                    },
                                },
                            },
                        },
                    }))
          .str());
  EXPECT_EQ(::mqt::test::errorStatus(
                [&] { return builtin_driver::addManifest(unavailable); }),
            QDMI_ERROR_LIBNOTFOUND);
  EXPECT_EQ(::mqt::test::errorStatus([&] {
              return builtin_driver::addManifest(directory.path() / "missing");
            }),
            QDMI_ERROR_LIBNOTFOUND);
  EXPECT_EQ(::mqt::test::errorStatus(
                [&] { return builtin_driver::addManifest(malformed); }),
            QDMI_ERROR_INVALIDARGUMENT);

  const ScopedEnvironmentVariable driver{"MQT_CORE_QDMI_DRIVER",
                                         MQT_CORE_QDMI_TEST_DRIVER};
  ::mqt::test::value(builtin_driver::addManifest(configured));
  ::mqt::test::value(builtin_driver::addManifest(configured));
  EXPECT_EQ(::mqt::test::errorStatus(
                [&] { return builtin_driver::addManifest(conflicting); }),
            QDMI_ERROR_INVALIDARGUMENT);
  ::mqt::test::value(builtin_driver::addManifest(unicode));
  {
    const ScopedEnvironmentVariable invalidConfig{"MQT_CORE_QDMI_CONFIG_JSON",
                                                  "{"};
    EXPECT_EQ(::mqt::test::errorStatus(
                  [&] { return builtin_driver::openDevice("test.session"); }),
              QDMI_ERROR_INVALIDARGUMENT);
  }
  // Failed driver initialization must leave manifest registration available.
  ::mqt::test::value(builtin_driver::addManifest(late));
  const auto first = [&] {
    const ScopedEnvironmentVariable overrides{
        "MQT_CORE_QDMI_CONFIG_JSON",
        R"({"schema-version":1,"qdmi":{"devices":[{"id":"test.session","session":{"custom4":"busy"}}]}})"};
    return ::mqt::test::value(builtin_driver::openDevice("test.session"));
  }();
  const auto second = ::mqt::test::value(
      builtin_driver::openDevice("test.session", R"({"custom4":"offline"})"));
  EXPECT_EQ(::mqt::test::value(first.getId()), "test.session");
  EXPECT_EQ(::mqt::test::value(second.getId()), "test.session");
  EXPECT_NE(first, second);
  EXPECT_EQ(::mqt::test::value(first.getStatus()), QDMI_DEVICE_STATUS_BUSY);
  EXPECT_EQ(::mqt::test::value(second.getStatus()), QDMI_DEVICE_STATUS_OFFLINE);
  EXPECT_EQ(::mqt::test::value(
                ::mqt::test::value(builtin_driver::openDevice("test.unicode"))
                    .getId()),
            "test.unicode");
  EXPECT_EQ(::mqt::test::errorStatus([&] {
              return builtin_driver::addManifest(
                  manifest("new.qdmi.json", "test.new"));
            }),
            QDMI_ERROR_BADSTATE);

  const ScopedEnvironmentVariable config{"QDMI_CONF",
                                         MQT_CORE_QDMI_EXAMPLE_CONFIG};
  auto replacement = ::mqt::test::value(
      Session::create(SessionConfig{.token = "replacement"}));
  EXPECT_EQ(::mqt::test::value(replacement.getDeviceIds()),
            std::vector<std::string>{"test.example"});
  EXPECT_EQ(::mqt::test::value(first.getStatus()), QDMI_DEVICE_STATUS_BUSY);
  EXPECT_THAT(
      ::mqt::test::errorMessage([] {
        return builtin_driver::openDevice(
            "unused", {},
            std::optional<std::filesystem::path>{MQT_CORE_QDMI_TEST_DRIVER});
      }),
      testing::HasSubstr("does not support targeted sessions"));
}

} // namespace
} // namespace qdmi
