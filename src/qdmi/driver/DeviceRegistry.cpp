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

#include "Driver.hpp"
#include "SessionConfig.hpp"

#include "qdmi/constants.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace qdmi::detail {
llvm::Error validateDeviceId(const std::string_view id) {
  if (id.empty()) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           "Device definition ID must not be empty");
  }
  if (id.find('\0') != std::string_view::npos) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           "Device definition ID must not contain NUL");
  }
  return llvm::Error::success();
}

namespace {
using Json = llvm::json::Value;

struct DefinitionPatch {
  std::string id;
  std::optional<std::filesystem::path> library;
  std::optional<std::string> prefix;
  std::optional<bool> enabled;
  DeviceSessionConfig session;
  std::string source;
};

struct DeviceManifestState {
  std::mutex mutex;
  std::vector<std::filesystem::path> paths;
  llvm::StringSet<> ids;
  bool frozen = false;
};

[[nodiscard]] auto deviceManifestState() -> DeviceManifestState& {
  static DeviceManifestState state;
  return state;
}

[[nodiscard]] auto sourceLabel(const std::string_view source,
                               const std::string_view path) -> std::string {
  return std::string(source) + ":" + std::string(path);
}

llvm::Error requireObject(const Json& value, const std::string_view source,
                          const std::string_view path) {
  if (value.getAsObject() == nullptr) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path) + " must be an object");
  }
  return llvm::Error::success();
}

llvm::Error
rejectUnknownKeys(const Json& value,
                  const std::initializer_list<std::string_view> allowed,
                  const std::string_view source, const std::string_view path) {
  for (const auto& [key, unused] : *value.getAsObject()) {
    static_cast<void>(unused);
    if (std::ranges::find(allowed, llvm::StringRef(key).str()) ==
        allowed.end()) {
      return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                             sourceLabel(source, path) +
                                 " contains unknown key '" +
                                 llvm::StringRef(key).str() + "'");
    }
  }
  return llvm::Error::success();
}

[[nodiscard]] auto optionalString(const Json& value, const std::string& key,
                                  const std::string_view source,
                                  const std::string& path)
    -> llvm::Expected<std::optional<std::string>> {
  const auto* const it = value.getAsObject()->get(key);
  if (it == nullptr) {
    return std::optional<std::string>{};
  }
  if (!it->getAsString()) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path + "." + key) +
                               " must be a string");
  }
  return std::optional<std::string>{it->getAsString()->str()};
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
parseSessionPatch(const Json& value, const std::string_view source,
                  const std::string& path, const std::filesystem::path& base)
    -> llvm::Expected<DeviceSessionConfig> {
  if (auto error = requireObject(value, source, path)) {
    return error;
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
    return error;
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
    if (!result) {
      return result.takeError();
    }
    *destination = std::move(*result);
  }
  if (const auto* const config = value.getAsObject()->get("device-config");
      config != nullptr) {
    const auto configPath = path + ".device-config";
    if (auto error = requireObject(*config, source, configPath)) {
      return error;
    }
    if (auto error = rejectUnknownKeys(*config, {"inline", "file"}, source,
                                       configPath)) {
      return error;
    }
    const auto* const inlineConfig = config->getAsObject()->get("inline");
    const auto* const fileConfig = config->getAsObject()->get("file");
    if ((inlineConfig == nullptr) == (fileConfig == nullptr)) {
      return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                             sourceLabel(source, configPath) +
                                 " must contain exactly one of 'inline' and "
                                 "'file'");
    }
    if (inlineConfig != nullptr) {
      if (inlineConfig->getAsObject() == nullptr) {
        return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                               sourceLabel(source, configPath + ".inline") +
                                   " must be an object");
      }
      patch.deviceConfiguration = InlineDeviceConfiguration{
          .json = llvm::formatv("{0}", *inlineConfig).str(),
      };
    } else {
      if (!fileConfig->getAsString() || fileConfig->getAsString()->empty() ||
          fileConfig->getAsString()->contains('\0')) {
        return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                               sourceLabel(source, configPath + ".file") +
                                   " must be a non-empty string");
      }
      auto filePath = pathFromString(*fileConfig->getAsString());
      if (!filePath) {
        return filePath.takeError();
      }
      patch.deviceConfiguration = FileDeviceConfiguration{
          .path = resolvePath(std::move(*filePath), base),
      };
    }
  }
  auto authFile = optionalString(value, "auth-file", source, path);
  if (!authFile) {
    return authFile.takeError();
  }
  if (*authFile) {
    if ((*authFile)->empty() || (*authFile)->find('\0') != std::string::npos) {
      return qdmi::makeError(
          QDMI_ERROR_INVALIDARGUMENT,
          sourceLabel(source, path + ".auth-file") +
              " must be a non-empty path without null bytes");
    }
    auto filePath = pathFromString(**authFile);
    if (!filePath) {
      return filePath.takeError();
    }
    patch.authFile = resolvePath(std::move(*filePath), base);
  }
  if (patch.deviceConfiguration && (patch.custom1 || patch.custom2)) {
    return qdmi::makeError(
        QDMI_ERROR_INVALIDARGUMENT,
        sourceLabel(source, path) +
            " must not combine device-config with custom1 or custom2");
  }
  return patch;
}

[[nodiscard]] auto
parseDevicePatch(const Json& value, const std::string_view source,
                 const std::string& path, const std::filesystem::path& base)
    -> llvm::Expected<DefinitionPatch> {
  if (auto error = requireObject(value, source, path)) {
    return error;
  }
  if (auto error = rejectUnknownKeys(
          value, {"id", "library", "prefix", "enabled", "session"}, source,
          path)) {
    return error;
  }
  auto idResult = optionalString(value, "id", source, path);
  if (!idResult) {
    return idResult.takeError();
  }
  const auto& id = (*idResult);
  if (!id || id->empty()) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path + ".id") +
                               " must be a non-empty string");
  }
  if (auto error = validateDeviceId(*id)) {
    return error;
  }
  DefinitionPatch patch;
  patch.id = *id;
  patch.source = source;
  auto library = optionalString(value, "library", source, path);
  if (!library) {
    return library.takeError();
  }
  if (*library) {
    if ((*library)->empty() || (*library)->find('\0') != std::string::npos) {
      return qdmi::makeError(
          QDMI_ERROR_INVALIDARGUMENT,
          sourceLabel(source, path + ".library") +
              " must be a non-empty path without null bytes");
    }
    auto libraryPath = pathFromString(**library);
    if (!libraryPath) {
      return libraryPath.takeError();
    }
    patch.library = resolvePath(std::move(*libraryPath), base);
  }
  auto prefix = optionalString(value, "prefix", source, path);
  if (!prefix) {
    return prefix.takeError();
  }
  patch.prefix = std::move(*prefix);
  if (patch.prefix && patch.prefix->find('\0') != std::string::npos) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path + ".prefix") +
                               " must not contain null bytes");
  }
  if (const auto* const it = value.getAsObject()->get("enabled");
      it != nullptr) {
    if (!it->getAsBoolean()) {
      return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                             sourceLabel(source, path + ".enabled") +
                                 " must be a boolean");
    }
    patch.enabled = it->getAsBoolean();
  }
  if (const auto* const it = value.getAsObject()->get("session");
      it != nullptr) {
    auto session = parseSessionPatch(*it, source, path + ".session", base);
    if (!session) {
      return session.takeError();
    }
    patch.session = std::move(*session);
  }
  return patch;
}

[[nodiscard]] auto parseConfiguration(const Json& root,
                                      const std::string_view source,
                                      const std::filesystem::path& base)
    -> llvm::Expected<std::vector<DefinitionPatch>> {
  if (auto error = requireObject(root, source, "$")) {
    return error;
  }
  if (auto error =
          rejectUnknownKeys(root, {"schema-version", "qdmi"}, source, "$")) {
    return error;
  }
  const auto* const version = root.getAsObject()->get("schema-version");
  if (version == nullptr || version->getAsUINT64() != 1) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, "$.schema-version") +
                               " must be the integer 1");
  }
  const auto* const qdmiConfig = root.getAsObject()->get("qdmi");
  if (qdmiConfig == nullptr) {
    return std::vector<DefinitionPatch>{};
  }
  if (auto error = requireObject(*qdmiConfig, source, "$.qdmi")) {
    return error;
  }
  if (auto error =
          rejectUnknownKeys(*qdmiConfig, {"devices"}, source, "$.qdmi")) {
    return error;
  }
  const auto* const devices = qdmiConfig->getAsObject()->get("devices");
  if (devices == nullptr) {
    return std::vector<DefinitionPatch>{};
  }
  if (devices->getAsArray() == nullptr) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, "$.qdmi.devices") +
                               " must be an array");
  }
  llvm::StringSet<> ids;
  std::vector<DefinitionPatch> patches;
  patches.reserve(devices->getAsArray()->size());
  for (size_t i = 0; i < devices->getAsArray()->size(); ++i) {
    auto result =
        parseDevicePatch((*devices->getAsArray())[i], source,
                         "$.qdmi.devices[" + std::to_string(i) + "]", base);
    if (!result) {
      return result.takeError();
    }
    auto& patch = (*result);
    if (!ids.insert(patch.id).second) {
      return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                             sourceLabel(source, "$.qdmi.devices") +
                                 " contains duplicate id '" + patch.id + "'");
    }
    patches.emplace_back(std::move(patch));
  }
  return patches;
}

[[nodiscard]] llvm::Expected<Json> parseJson(std::string_view text,
                                             const std::string_view source) {
  auto value = llvm::json::parse(text);
  if (!value) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           std::string(source) + ": " +
                               llvm::toString(value.takeError()));
  }
  return std::move(*value);
}

[[nodiscard]] llvm::Expected<Json> readJson(const std::filesystem::path& path) {
  auto source = pathToString(path);
  if (!source) {
    return source.takeError();
  }
  auto buffer = llvm::MemoryBuffer::getFile(*source);
  if (!buffer) {
    return qdmi::makeError(QDMI_ERROR_FATAL,
                           *source + ": " + buffer.getError().message());
  }
  return parseJson((*buffer)->getBuffer(), *source);
}

void mergePatch(DefinitionPatch& target, const DefinitionPatch& source) {
  applyOverride(target.library, source.library);
  applyOverride(target.prefix, source.prefix);
  applyOverride(target.enabled, source.enabled);
  target.session = mergeSessionConfig(target.session, source.session);
  target.source = source.source;
}

llvm::Error appendIfFile(std::vector<std::filesystem::path>& files,
                         const std::filesystem::path& path) {
  const auto& absolute = path;
  if (absolute.empty()) {
    return llvm::Error::success();
  }
  auto filename = pathToString(absolute);
  if (!filename) {
    return filename.takeError();
  }
  if (llvm::sys::fs::is_regular_file(*filename)) {
    files.emplace_back(absolute);
  }
  return llvm::Error::success();
}

llvm::Error appendFragments(std::vector<std::filesystem::path>& files,
                            const std::filesystem::path& directory) {
  const auto& absolute = directory;
  if (absolute.empty()) {
    return llvm::Error::success();
  }
  auto filename = pathToString(absolute);
  if (!filename) {
    return filename.takeError();
  }
  if (!llvm::sys::fs::is_directory(*filename)) {
    return llvm::Error::success();
  }
  std::vector<std::filesystem::path> found;
  std::error_code error;
  for (llvm::sys::fs::directory_iterator entry(*filename, error), end;
       !error && entry != end; entry.increment(error)) {
    const auto status = entry->status();
    if (!status) {
      return qdmi::makeError(QDMI_ERROR_FATAL, entry->path() + ": " +
                                                   status.getError().message());
    }
    if (llvm::sys::fs::is_regular_file(*status) &&
        llvm::StringRef(entry->path()).ends_with(".qdmi.json")) {
      auto foundPath = pathFromString(entry->path());
      if (!foundPath) {
        return foundPath.takeError();
      }
      found.emplace_back(std::move(*foundPath));
    }
  }
  if (error) {
    return qdmi::makeError(QDMI_ERROR_FATAL,
                           *filename + ": " + error.message());
  }
  std::ranges::sort(found);
  files.insert(files.end(), found.begin(), found.end());
  return llvm::Error::success();
}

[[nodiscard]] auto nearestProjectConfiguration(std::filesystem::path directory)
    -> llvm::Expected<std::optional<std::filesystem::path>> {
  while (!directory.empty()) {
    auto dedicated = directory / "qdmi.json";
    auto filename = pathToString(dedicated);
    if (!filename) {
      return filename.takeError();
    }
    bool regular = false;
    const auto error = llvm::sys::fs::is_regular_file(*filename, regular);
    if (regular) {
      return std::optional<std::filesystem::path>{std::move(dedicated)};
    }
    if (error && error != std::errc::no_such_file_or_directory &&
        error != std::errc::not_a_directory) {
      return qdmi::makeError(QDMI_ERROR_FATAL,
                             *filename + ": " + error.message());
    }
    const auto parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return std::optional<std::filesystem::path>{};
}

[[nodiscard]] auto discoverFiles(const std::filesystem::path& cwd)
    -> llvm::Expected<std::vector<std::filesystem::path>> {
  std::vector<std::filesystem::path> files;
  auto module = moduleDirectory(reinterpret_cast<const void*>(&discoverFiles));
  if (!module) {
    return module.takeError();
  }
  const auto root = resolvePath(*module, cwd);
  for (const auto& directory : {
           root,
           root / "bin",
           root / "lib",
           root / "mqt-core" / "qdmi",
           root / "qdmi",
       }) {
    if (auto error = appendFragments(files, directory)) {
      return error;
    }
  }

  std::optional<std::filesystem::path> explicitFile;
  if (auto value = llvm::sys::Process::GetEnv("MQT_CORE_QDMI_CONFIG_FILE")) {
    auto filename = pathFromString(*value);
    if (!filename) {
      return filename.takeError();
    }
    explicitFile = std::move(*filename);
  }
  if (explicitFile) {
    const auto resolved = resolvePath(*explicitFile, cwd);
    auto filename = pathToString(resolved);
    if (!filename) {
      return filename.takeError();
    }
    if (!llvm::sys::fs::is_regular_file(*filename)) {
      return qdmi::makeError(QDMI_ERROR_FATAL,
                             "Explicit QDMI configuration file does not "
                             "exist: " +
                                 *filename);
    }
    files.emplace_back(resolved);
    return files;
  }

#ifdef _WIN32
  for (const auto* variable : {"PROGRAMDATA", "APPDATA"}) {
    if (auto value = llvm::sys::Process::GetEnv(variable)) {
      auto directory = pathFromString(*value);
      if (!directory) {
        return directory.takeError();
      }
      if (auto error =
              appendIfFile(files, *directory / "mqt-core" / "qdmi.json")) {
        return error;
      }
    }
  }
#else
  if (auto error = appendIfFile(files, "/etc/mqt-core/qdmi.json")) {
    return error;
  }
  if (auto xdg = llvm::sys::Process::GetEnv("XDG_CONFIG_HOME");
      xdg && !xdg->empty()) {
    if (auto error = appendIfFile(files, std::filesystem::path(*xdg) /
                                             "mqt-core" / "qdmi.json")) {
      return error;
    }
  } else if (auto home = llvm::sys::Process::GetEnv("HOME")) {
    if (auto error =
            appendIfFile(files, std::filesystem::path(*home) / ".config" /
                                    "mqt-core" / "qdmi.json")) {
      return error;
    }
  }
#endif
  auto project = nearestProjectConfiguration(cwd);
  if (!project) {
    return project.takeError();
  }
  if ((*project)) {
    files.emplace_back(*(*project));
  }
  for (auto& file : files) {
    file = resolvePath(std::move(file), cwd);
  }
  return files;
}

[[nodiscard]] auto materialize(const DefinitionPatch& patch)
    -> llvm::Expected<qdmi::DeviceDefinition> {
  if (!patch.library || patch.library->empty()) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           patch.source + ": enabled device '" + patch.id +
                               "' is missing library");
  }
  if (!patch.prefix || patch.prefix->empty()) {
    return qdmi::makeError(QDMI_ERROR_INVALIDARGUMENT,
                           patch.source + ": enabled device '" + patch.id +
                               "' is missing prefix");
  }
  qdmi::DeviceDefinition definition;
  definition.id = patch.id;
  definition.library = *patch.library;
  definition.prefix = *patch.prefix;
  definition.session = patch.session;
  return definition;
}

} // namespace

auto stageDeviceManifest(const std::filesystem::path& path) -> int {
  if (path.empty()) {
    return QDMI_ERROR_INVALIDARGUMENT;
  }

  std::error_code error;
  const auto canonical = std::filesystem::weakly_canonical(path, error);
  if (error) {
    return QDMI_ERROR_LIBNOTFOUND;
  }

  auto& state = deviceManifestState();
  {
    const std::scoped_lock lock(state.mutex);
    if (std::ranges::find(state.paths, canonical) != state.paths.end()) {
      return QDMI_SUCCESS;
    }
    if (state.frozen) {
      return QDMI_ERROR_BADSTATE;
    }
  }
  auto filename = pathToString(canonical);
  if (!filename) {
    return qdmi::toStatus(filename.takeError());
  }
  if (!llvm::sys::fs::is_regular_file(*filename)) {
    return QDMI_ERROR_LIBNOTFOUND;
  }

  auto root = readJson(canonical);
  if (!root) {
    return qdmi::toStatus(root.takeError());
  }
  auto patches = parseConfiguration(*root, *filename, canonical.parent_path());
  if (!patches) {
    return qdmi::toStatus(patches.takeError());
  }
  for (const auto& patch : *patches) {
    if (!patch.enabled.value_or(true)) {
      continue;
    }
    auto definition = materialize(patch);
    if (!definition) {
      return qdmi::toStatus(definition.takeError());
    }
    bool regular = false;
    auto libraryPath = pathToString(definition->library);
    if (!libraryPath) {
      return qdmi::toStatus(libraryPath.takeError());
    }
    error = llvm::sys::fs::is_regular_file(*libraryPath, regular);
    if (error && error != std::errc::no_such_file_or_directory &&
        error != std::errc::not_a_directory) {
      return QDMI_ERROR_FATAL;
    }
    if (!regular) {
      return QDMI_ERROR_LIBNOTFOUND;
    }
  }

  const std::scoped_lock lock(state.mutex);
  if (std::ranges::find(state.paths, canonical) != state.paths.end()) {
    return QDMI_SUCCESS;
  }
  if (state.frozen) {
    return QDMI_ERROR_BADSTATE;
  }
  for (const auto& patch : *patches) {
    if (state.ids.contains(patch.id)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
  }
  state.paths.emplace_back(canonical);
  for (const auto& patch : *patches) {
    state.ids.insert(patch.id);
  }
  return QDMI_SUCCESS;
}

auto parseDeviceSessionJson(const char* const data, const size_t size,
                            DeviceSessionConfig& config) -> int {
  config = {};
  if (data == nullptr && size == 0) {
    return QDMI_SUCCESS;
  }

  auto value = parseJson(std::string_view{data, size}, "<device-session-json>");
  if (!value) {
    return qdmi::toStatus(value.takeError());
  }
  llvm::SmallString<256> cwd;
  if (llvm::sys::fs::current_path(cwd)) {
    return QDMI_ERROR_FATAL;
  }
  auto directory = pathFromString(cwd.str());
  if (!directory) {
    return qdmi::toStatus(directory.takeError());
  }
  auto parsed =
      parseSessionPatch(*value, "<device-session-json>", "$", *directory);
  if (!parsed) {
    return qdmi::toStatus(parsed.takeError());
  }
  config = std::move(*parsed);
  return QDMI_SUCCESS;
}

auto freezeDeviceManifests() -> std::vector<std::filesystem::path> {
  auto& state = deviceManifestState();
  const std::scoped_lock lock(state.mutex);
  state.frozen = true;
  return state.paths;
}

void rollbackDeviceManifestFreeze() {
  auto& state = deviceManifestState();
  const std::scoped_lock lock(state.mutex);
  state.frozen = false;
}

llvm::Expected<DeviceRegistry> DeviceRegistry::discover() {
  llvm::SmallString<256> directory;
  if (const auto filesystemError = llvm::sys::fs::current_path(directory)) {
    return qdmi::makeError(QDMI_ERROR_FATAL, "Cannot read current directory: " +
                                                 filesystemError.message());
  }
  auto cwd = pathFromString(directory.str());
  if (!cwd) {
    return cwd.takeError();
  }
  auto files = discoverFiles(*cwd);
  if (!files) {
    return files.takeError();
  }
  std::map<std::string, DefinitionPatch> merged;
  const auto mergePatches =
      [&merged](const Json& root, const std::string_view source,
                const std::filesystem::path& base) -> llvm::Error {
    auto result = parseConfiguration(root, source, base);
    if (!result) {
      return result.takeError();
    }
    for (auto& patch : (*result)) {
      if (auto const it = merged.find(patch.id); it != merged.end()) {
        mergePatch(it->second, patch);
      } else {
        merged.emplace(patch.id, std::move(patch));
      }
    }
    return llvm::Error::success();
  };
  auto staged = freezeDeviceManifests();
  files->insert(files->begin(), staged.begin(), staged.end());
  for (const auto& file : (*files)) {
    auto root = readJson(file);
    if (!root) {
      return root.takeError();
    }
    auto source = pathToString(file);
    if (!source) {
      return source.takeError();
    }
    if (auto error = mergePatches((*root), *source, file.parent_path())) {
      return error;
    }
  }
  if (auto inlineJson =
          llvm::sys::Process::GetEnv("MQT_CORE_QDMI_CONFIG_JSON")) {
    auto root = parseJson(*inlineJson, "<MQT_CORE_QDMI_CONFIG_JSON>");
    if (!root) {
      return root.takeError();
    }
    if (auto error =
            mergePatches((*root), "<MQT_CORE_QDMI_CONFIG_JSON>", *cwd)) {
      return error;
    }
  }
  DeviceRegistry registry;
  for (auto& [unused, patch] : merged) {
    if (!patch.enabled.value_or(true)) {
      registry.disabledIds_.emplace_back(std::move(patch.id));
    } else {
      auto definition = materialize(patch);
      if (!definition) {
        return definition.takeError();
      }
      registry.definitions_.emplace_back(std::move(*definition));
    }
  }
  return registry;
}

} // namespace qdmi::detail
