/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "DeviceRegistry.hpp"

#include "qdmi/common/Common.hpp"
#include "qdmi/common/DeviceConfiguration.hpp"
#include "qdmi/driver/Driver.hpp"
#include "qdmi/driver/SessionConfig.hpp"

#include "JSON.hpp"

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "qdmi/constants.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace qdmi::detail {
namespace {
using Json = nlohmann::json;

struct DefinitionPatch {
  std::string id;
  std::optional<std::filesystem::path> library;
  std::optional<std::string> prefix;
  std::optional<bool> enabled;
  DeviceSessionConfig session;
  std::filesystem::path source;
};

[[nodiscard]] auto sourceLabel(const std::filesystem::path& source,
                               const std::string_view path) -> std::string {
  return source.string() + ":" + std::string(path);
}

std::optional<Error> requireObject(const Json& value,
                                   const std::filesystem::path& source,
                                   const std::string_view path) {
  if (!value.is_object()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = sourceLabel(source, path) + " must be an object",
    };
  }
  return std::nullopt;
}

std::optional<Error> rejectUnknownKeys(
    const Json& value, const std::initializer_list<std::string_view> allowed,
    const std::filesystem::path& source, const std::string_view path) {
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (std::ranges::find(allowed, key) == allowed.end()) {
      return Error{
          .status = QDMI_ERROR_INVALIDARGUMENT,
          .message =
              sourceLabel(source, path) + " contains unknown key '" + key + "'",
      };
    }
  }
  return std::nullopt;
}

[[nodiscard]] auto optionalString(const Json& value, const std::string& key,
                                  const std::filesystem::path& source,
                                  const std::string& path)
    -> Result<std::optional<std::string>> {
  const auto it = value.find(key);
  if (it == value.end()) {
    return std::nullopt;
  }
  if (!it->is_string()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = sourceLabel(source, path + "." + key) + " must be a string",
    };
  }
  return it->get<std::string>();
}

[[nodiscard]] auto resolvePath(std::filesystem::path path,
                               const std::filesystem::path& base)
    -> std::filesystem::path {
  if (path.is_relative()) {
    path = base / path;
  }
  return path.lexically_normal();
}

[[nodiscard]] auto
parseSessionPatch(const Json& value, const std::filesystem::path& source,
                  const std::string& path, const std::filesystem::path& base)
    -> Result<DeviceSessionConfig> {
  if (auto error = requireObject(value, source, path)) {
    return std::move(*error);
  }
  if (auto error = rejectUnknownKeys(value,
                                     {
                                         "base-url",
                                         "token",
                                         "auth-file",
                                         "auth-url",
                                         "username",
                                         "password",
                                         "custom1",
                                         "custom2",
                                         "custom3",
                                         "custom4",
                                         "custom5",
                                         "device-config",
                                     },
                                     source, path)) {
    return std::move(*error);
  }
  DeviceSessionConfig patch;
  const std::array<std::pair<const char*, std::optional<std::string>*>, 10>
      fields = {
          {
              {"base-url", &patch.baseUrl},
              {"token", &patch.token},
              {"auth-url", &patch.authUrl},
              {"username", &patch.username},
              {"password", &patch.password},
              {"custom1", &patch.custom1},
              {"custom2", &patch.custom2},
              {"custom3", &patch.custom3},
              {"custom4", &patch.custom4},
              {"custom5", &patch.custom5},
          },
  };
  for (const auto& [key, destination] : fields) {
    auto result = optionalString(value, key, source, path);
    if (auto* error = std::get_if<Error>(&result)) {
      return std::move(*error);
    }
    *destination = std::get<0>(std::move(result));
  }
  if (const auto config = value.find("device-config"); config != value.end()) {
    const auto configPath = path + ".device-config";
    if (auto error = requireObject(*config, source, configPath)) {
      return std::move(*error);
    }
    if (auto error = rejectUnknownKeys(*config, {"inline", "file"}, source,
                                       configPath)) {
      return std::move(*error);
    }
    const auto inlineConfig = config->find("inline");
    const auto fileConfig = config->find("file");
    if ((inlineConfig == config->end()) == (fileConfig == config->end())) {
      return Error{
          .status = QDMI_ERROR_INVALIDARGUMENT,
          .message = sourceLabel(source, configPath) +
                     " must contain exactly one of 'inline' and "
                     "'file'",
      };
    }
    if (inlineConfig != config->end()) {
      if (!inlineConfig->is_object()) {
        return Error{
            .status = QDMI_ERROR_INVALIDARGUMENT,
            .message = sourceLabel(source, configPath + ".inline") +
                       " must be an object",
        };
      }
      patch.deviceConfiguration =
          InlineDeviceConfiguration{.json = inlineConfig->dump()};
    } else {
      if (!fileConfig->is_string() ||
          fileConfig->get_ref<const std::string&>().empty()) {
        return Error{
            .status = QDMI_ERROR_INVALIDARGUMENT,
            .message = sourceLabel(source, configPath + ".file") +
                       " must be a non-empty string",
        };
      }
      patch.deviceConfiguration = FileDeviceConfiguration{
          .path = resolvePath(fileConfig->get<std::string>(), base),
      };
    }
  }
  auto authFile = optionalString(value, "auth-file", source, path);
  if (auto* error = std::get_if<Error>(&authFile)) {
    return std::move(*error);
  }
  if (std::get<0>(authFile)) {
    patch.authFile = resolvePath(*std::get<0>(authFile), base);
  }
  return patch;
}

[[nodiscard]] auto
parseDevicePatch(const Json& value, const std::filesystem::path& source,
                 const std::string& path, const std::filesystem::path& base)
    -> Result<DefinitionPatch> {
  if (auto error = requireObject(value, source, path)) {
    return std::move(*error);
  }
  if (auto error = rejectUnknownKeys(
          value, {"id", "library", "prefix", "enabled", "session"}, source,
          path)) {
    return std::move(*error);
  }
  auto idResult = optionalString(value, "id", source, path);
  if (auto* error = std::get_if<Error>(&idResult)) {
    return std::move(*error);
  }
  const auto& id = std::get<0>(idResult);
  if (!id || id->empty()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message =
            sourceLabel(source, path + ".id") + " must be a non-empty string",
    };
  }
  DefinitionPatch patch;
  patch.id = *id;
  patch.source = source;
  auto library = optionalString(value, "library", source, path);
  if (auto* error = std::get_if<Error>(&library)) {
    return std::move(*error);
  }
  if (std::get<0>(library)) {
    patch.library = resolvePath(*std::get<0>(library), base);
  }
  auto prefix = optionalString(value, "prefix", source, path);
  if (auto* error = std::get_if<Error>(&prefix)) {
    return std::move(*error);
  }
  patch.prefix = std::get<0>(std::move(prefix));
  if (const auto it = value.find("enabled"); it != value.end()) {
    if (!it->is_boolean()) {
      return Error{
          .status = QDMI_ERROR_INVALIDARGUMENT,
          .message =
              sourceLabel(source, path + ".enabled") + " must be a boolean",
      };
    }
    patch.enabled = it->get<bool>();
  }
  if (const auto it = value.find("session"); it != value.end()) {
    auto session = parseSessionPatch(*it, source, path + ".session", base);
    if (auto* error = std::get_if<Error>(&session)) {
      return std::move(*error);
    }
    patch.session = std::get<0>(std::move(session));
  }
  return patch;
}

[[nodiscard]] auto parseConfiguration(const Json& root,
                                      const std::filesystem::path& source,
                                      const std::filesystem::path& base)
    -> Result<std::vector<DefinitionPatch>> {
  if (auto error = requireObject(root, source, "$")) {
    return std::move(*error);
  }
  if (auto error =
          rejectUnknownKeys(root, {"schema-version", "qdmi"}, source, "$")) {
    return std::move(*error);
  }
  const auto version = root.find("schema-version");
  if (version == root.end() || !version->is_number_integer() || *version != 1) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message =
            sourceLabel(source, "$.schema-version") + " must be the integer 1",
    };
  }
  const auto qdmiConfig = root.find("qdmi");
  if (qdmiConfig == root.end()) {
    return {};
  }
  if (auto error = requireObject(*qdmiConfig, source, "$.qdmi")) {
    return std::move(*error);
  }
  if (auto error =
          rejectUnknownKeys(*qdmiConfig, {"devices"}, source, "$.qdmi")) {
    return std::move(*error);
  }
  const auto devices = qdmiConfig->find("devices");
  if (devices == qdmiConfig->end()) {
    return {};
  }
  if (!devices->is_array()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = sourceLabel(source, "$.qdmi.devices") + " must be an array",
    };
  }
  std::set<std::string> ids;
  std::vector<DefinitionPatch> patches;
  patches.reserve(devices->size());
  for (size_t i = 0; i < devices->size(); ++i) {
    auto result =
        parseDevicePatch((*devices)[i], source,
                         "$.qdmi.devices[" + std::to_string(i) + "]", base);
    if (auto* error = std::get_if<Error>(&result)) {
      return std::move(*error);
    }
    auto& patch = std::get<0>(result);
    if (!ids.emplace(patch.id).second) {
      return Error{
          .status = QDMI_ERROR_INVALIDARGUMENT,
          .message = sourceLabel(source, "$.qdmi.devices") +
                     " contains duplicate id '" + patch.id + "'",
      };
    }
    patches.emplace_back(std::move(patch));
  }
  return patches;
}

[[nodiscard]] Result<Json> parseJson(std::string_view text,
                                     const std::filesystem::path& source) {
  auto result = mqt::detail::parseJSON(text);
  if (auto const* error = std::get_if<std::string>(&result)) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = source.string() + ": invalid JSON: " + *error,
    };
  }
  return std::get<0>(std::move(result));
}

[[nodiscard]] Result<Json> readJson(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    return Error{
        .status = QDMI_ERROR_NOTFOUND,
        .message = "Cannot open QDMI configuration file: " + path.string(),
    };
  }
  const std::string text{std::istreambuf_iterator<char>(stream), {}};
  if (stream.bad()) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message = "Cannot read QDMI configuration file: " + path.string(),
    };
  }
  return parseJson(text, path);
}

void mergePatch(DefinitionPatch& target, const DefinitionPatch& source) {
  applyOverride(target.library, source.library);
  applyOverride(target.prefix, source.prefix);
  applyOverride(target.enabled, source.enabled);
  target.session = mergeSessionConfig(target.session, source.session);
  target.source = source.source;
}

void appendIfFile(std::vector<std::filesystem::path>& files,
                  const std::filesystem::path& path) {
  const auto& absolute = path;
  if (absolute.empty()) {
    return;
  }
  std::error_code error;
  if (std::filesystem::is_regular_file(absolute, error)) {
    files.emplace_back(absolute);
  }
}

std::optional<Error> appendFragments(std::vector<std::filesystem::path>& files,
                                     const std::filesystem::path& directory) {
  const auto& absolute = directory;
  if (absolute.empty()) {
    return std::nullopt;
  }
  std::error_code error;
  if (!std::filesystem::is_directory(absolute, error)) {
    return std::nullopt;
  }
  std::vector<std::filesystem::path> found;
  for (std::filesystem::directory_iterator entry(absolute, error), end;
       !error && entry != end; entry.increment(error)) {
    const auto regular = entry->is_regular_file(error);
    if (error) {
      return Error{
          .status = QDMI_ERROR_FATAL,
          .message = entry->path().string() + ": " + error.message(),
      };
    }
    if (regular && entry->path().filename().string().ends_with(".qdmi.json")) {
      found.emplace_back(entry->path());
    }
  }
  if (error) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message = absolute.string() + ": " + error.message(),
    };
  }
  std::ranges::sort(found);
  files.insert(files.end(), found.begin(), found.end());
  return std::nullopt;
}

[[nodiscard]] auto nearestProjectConfiguration(std::filesystem::path directory)
    -> Result<std::optional<std::filesystem::path>> {
  while (!directory.empty()) {
    auto dedicated = directory / "qdmi.json";
    std::error_code error;
    if (std::filesystem::is_regular_file(dedicated, error)) {
      return dedicated;
    }
    if (error && error != std::errc::no_such_file_or_directory &&
        error != std::errc::not_a_directory) {
      return Error{
          .status = QDMI_ERROR_FATAL,
          .message = dedicated.string() + ": " + error.message(),
      };
    }
    const auto parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return std::nullopt;
}

[[nodiscard]] auto discoverFiles(const std::filesystem::path& cwd)
    -> Result<std::vector<std::filesystem::path>> {
  std::vector<std::filesystem::path> files;
  const auto root = resolvePath(
      moduleDirectory(reinterpret_cast<const void*>(&discoverFiles)), cwd);
  for (const auto& directory : {
           root,
           root / "bin",
           root / "lib",
           root / "mqt-core" / "qdmi",
           root / "qdmi",
       }) {
    if (auto error = appendFragments(files, directory)) {
      return std::move(*error);
    }
  }

  std::optional<std::filesystem::path> explicitFile;
  if (auto value = environment("MQT_CORE_QDMI_CONFIG_FILE")) {
    explicitFile = *value;
  }
  if (explicitFile) {
    const auto resolved = resolvePath(*explicitFile, cwd);
    std::error_code error;
    if (!std::filesystem::is_regular_file(resolved, error)) {
      return Error{
          .status = QDMI_ERROR_FATAL,
          .message = "Explicit QDMI configuration file does not "
                     "exist: " +
                     resolved.string(),
      };
    }
    files.emplace_back(resolved);
    return files;
  }

#ifdef _WIN32
  if (auto programData = environment("PROGRAMDATA")) {
    appendIfFile(files, std::filesystem::path(*programData) / "mqt-core" /
                            "qdmi.json");
  }
  if (auto appData = environment("APPDATA")) {
    appendIfFile(files,
                 std::filesystem::path(*appData) / "mqt-core" / "qdmi.json");
  }
#else
  appendIfFile(files, "/etc/mqt-core/qdmi.json");
  if (auto xdg = environment("XDG_CONFIG_HOME")) {
    appendIfFile(files, std::filesystem::path(*xdg) / "mqt-core" / "qdmi.json");
  } else if (auto home = environment("HOME")) {
    appendIfFile(files, std::filesystem::path(*home) / ".config" / "mqt-core" /
                            "qdmi.json");
  }
#endif
  auto project = nearestProjectConfiguration(cwd);
  if (auto* error = std::get_if<Error>(&project)) {
    return std::move(*error);
  }
  if (std::get<0>(project)) {
    files.emplace_back(*std::get<0>(project));
  }
  for (auto& file : files) {
    file = resolvePath(std::move(file), cwd);
  }
  return files;
}

[[nodiscard]] auto materialize(const DefinitionPatch& patch)
    -> Result<qdmi::DeviceDefinition> {
  if (!patch.library || patch.library->empty()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = patch.source.string() + ": enabled device '" + patch.id +
                   "' is missing library",
    };
  }
  if (!patch.prefix || patch.prefix->empty()) {
    return Error{
        .status = QDMI_ERROR_INVALIDARGUMENT,
        .message = patch.source.string() + ": enabled device '" + patch.id +
                   "' is missing prefix",
    };
  }
  qdmi::DeviceDefinition definition;
  definition.id = patch.id;
  definition.library = *patch.library;
  definition.prefix = *patch.prefix;
  definition.session = patch.session;
  return definition;
}

} // namespace

Result<DeviceRegistry> DeviceRegistry::discover() {
  std::error_code filesystemError;
  const auto cwd = std::filesystem::current_path(filesystemError);
  if (filesystemError) {
    return Error{
        .status = QDMI_ERROR_FATAL,
        .message =
            "Cannot read current directory: " + filesystemError.message(),
    };
  }
  auto files = discoverFiles(cwd);
  if (auto* error = std::get_if<Error>(&files)) {
    return std::move(*error);
  }
  std::map<std::string, DefinitionPatch> merged;
  const auto mergePatches =
      [&merged](const Json& root, const std::filesystem::path& source,
                const std::filesystem::path& base) -> std::optional<Error> {
    auto result = parseConfiguration(root, source, base);
    if (auto* error = std::get_if<Error>(&result)) {
      return std::move(*error);
    }
    for (auto& patch : std::get<0>(result)) {
      if (auto const it = merged.find(patch.id); it != merged.end()) {
        mergePatch(it->second, patch);
      } else {
        merged.emplace(patch.id, std::move(patch));
      }
    }
    return std::nullopt;
  };
  for (const auto& file : std::get<0>(files)) {
    auto root = readJson(file);
    if (auto* error = std::get_if<Error>(&root)) {
      return std::move(*error);
    }
    if (auto error =
            mergePatches(std::get<0>(root), file, file.parent_path())) {
      return std::move(*error);
    }
  }
  if (auto inlineJson = environment("MQT_CORE_QDMI_CONFIG_JSON")) {
    auto root = parseJson(*inlineJson, "<MQT_CORE_QDMI_CONFIG_JSON>");
    if (auto* error = std::get_if<Error>(&root)) {
      return std::move(*error);
    }
    if (auto error = mergePatches(std::get<0>(root),
                                  "<MQT_CORE_QDMI_CONFIG_JSON>", cwd)) {
      return std::move(*error);
    }
  }
  DeviceRegistry registry;
  for (auto& [unused, patch] : merged) {
    if (!patch.enabled.value_or(true)) {
      registry.disabledIds_.emplace_back(std::move(patch.id));
    } else {
      auto definition = materialize(patch);
      if (auto* error = std::get_if<Error>(&definition)) {
        return std::move(*error);
      }
      registry.definitions_.emplace_back(std::get<0>(std::move(definition)));
    }
  }
  return registry;
}

} // namespace qdmi::detail
