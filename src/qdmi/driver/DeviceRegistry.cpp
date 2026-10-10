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
#include "JSON.hpp"
#include "SessionConfig.hpp"

#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"
#include "qdmi/constants.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Process.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace qdmi::detail {
llvm::LogicalResult validateDeviceId(const std::string_view id) {
  if (id.empty()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           "Device definition ID must not be empty");
  }
  if (id.find('\0') != std::string_view::npos) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           "Device definition ID must not contain NUL");
  }
  return llvm::success();
}

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

struct DeviceManifestState {
  std::mutex mutex;
  std::vector<std::filesystem::path> paths;
  std::set<std::string> ids;
  bool frozen = false;
};

[[nodiscard]] auto deviceManifestState() -> DeviceManifestState& {
  static DeviceManifestState state;
  return state;
}

[[nodiscard]] auto sourceLabel(const std::filesystem::path& source,
                               const std::string_view path) -> std::string {
  return pathToString(source) + ":" + std::string(path);
}

llvm::LogicalResult requireObject(const Json& value,
                                  const std::filesystem::path& source,
                                  const std::string_view path) {
  if (!value.is_object()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path) + " must be an object");
  }
  return llvm::success();
}

llvm::LogicalResult rejectUnknownKeys(
    const Json& value, const std::initializer_list<std::string_view> allowed,
    const std::filesystem::path& source, const std::string_view path) {
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (std::ranges::find(allowed, key) == allowed.end()) {
      return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                             sourceLabel(source, path) +
                                 " contains unknown key '" + key + "'");
    }
  }
  return llvm::success();
}

[[nodiscard]] auto optionalString(const Json& value, const std::string& key,
                                  const std::filesystem::path& source,
                                  const std::string& path)
    -> llvm::FailureOr<std::optional<std::string>> {
  const auto it = value.find(key);
  if (it == value.end()) {
    return std::optional<std::string>{};
  }
  if (!it->is_string()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path + "." + key) +
                               " must be a string");
  }
  return std::optional<std::string>{std::move(it->get<std::string>())};
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
    -> llvm::FailureOr<DeviceSessionConfig> {
  if (llvm::failed(requireObject(value, source, path))) {
    return llvm::failure();
  }
  if (llvm::failed(rejectUnknownKeys(value,
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
                                     source, path))) {
    return llvm::failure();
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
    if (llvm::failed(result)) {
      return llvm::failure();
    }
    *destination = (*std::move(result));
  }
  if (const auto config = value.find("device-config"); config != value.end()) {
    const auto configPath = path + ".device-config";
    if (llvm::failed(requireObject(*config, source, configPath))) {
      return llvm::failure();
    }
    if (llvm::failed(rejectUnknownKeys(*config, {"inline", "file"}, source,
                                       configPath))) {
      return llvm::failure();
    }
    const auto inlineConfig = config->find("inline");
    const auto fileConfig = config->find("file");
    if ((inlineConfig == config->end()) == (fileConfig == config->end())) {
      return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                             sourceLabel(source, configPath) +
                                 " must contain exactly one of 'inline' and "
                                 "'file'");
    }
    if (inlineConfig != config->end()) {
      if (!inlineConfig->is_object()) {
        return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                               sourceLabel(source, configPath + ".inline") +
                                   " must be an object");
      }
      patch.deviceConfiguration =
          InlineDeviceConfiguration{.json = inlineConfig->dump()};
    } else {
      if (!fileConfig->is_string() ||
          fileConfig->get_ref<const std::string&>().empty() ||
          fileConfig->get_ref<const std::string&>().find('\0') !=
              std::string::npos) {
        return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                               sourceLabel(source, configPath + ".file") +
                                   " must be a non-empty string");
      }
      patch.deviceConfiguration = FileDeviceConfiguration{
          .path = resolvePath(
              pathFromString(fileConfig->get_ref<const std::string&>()), base),
      };
    }
  }
  auto authFile = optionalString(value, "auth-file", source, path);
  if (llvm::failed(authFile)) {
    return llvm::failure();
  }
  if (*authFile) {
    if ((*authFile)->empty() || (*authFile)->find('\0') != std::string::npos) {
      return qdmi::emitError(
          QDMI_ERROR_INVALIDARGUMENT,
          sourceLabel(source, path + ".auth-file") +
              " must be a non-empty path without null bytes");
    }
    patch.authFile = resolvePath(pathFromString(**authFile), base);
  }
  if (patch.deviceConfiguration && (patch.custom1 || patch.custom2)) {
    return qdmi::emitError(
        QDMI_ERROR_INVALIDARGUMENT,
        sourceLabel(source, path) +
            " must not combine device-config with custom1 or custom2");
  }
  return patch;
}

[[nodiscard]] auto
parseDevicePatch(const Json& value, const std::filesystem::path& source,
                 const std::string& path, const std::filesystem::path& base)
    -> llvm::FailureOr<DefinitionPatch> {
  if (llvm::failed(requireObject(value, source, path))) {
    return llvm::failure();
  }
  if (llvm::failed(rejectUnknownKeys(
          value, {"id", "library", "prefix", "enabled", "session"}, source,
          path))) {
    return llvm::failure();
  }
  auto idResult = optionalString(value, "id", source, path);
  if (llvm::failed(idResult)) {
    return llvm::failure();
  }
  const auto& id = (*idResult);
  if (!id || id->empty()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path + ".id") +
                               " must be a non-empty string");
  }
  if (llvm::failed(validateDeviceId(*id))) {
    return llvm::failure();
  }
  DefinitionPatch patch;
  patch.id = *id;
  patch.source = source;
  auto library = optionalString(value, "library", source, path);
  if (llvm::failed(library)) {
    return llvm::failure();
  }
  if (*library) {
    if ((*library)->empty() || (*library)->find('\0') != std::string::npos) {
      return qdmi::emitError(
          QDMI_ERROR_INVALIDARGUMENT,
          sourceLabel(source, path + ".library") +
              " must be a non-empty path without null bytes");
    }
    patch.library = resolvePath(pathFromString(**library), base);
  }
  auto prefix = optionalString(value, "prefix", source, path);
  if (llvm::failed(prefix)) {
    return llvm::failure();
  }
  patch.prefix = (*std::move(prefix));
  if (patch.prefix && patch.prefix->find('\0') != std::string::npos) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path + ".prefix") +
                               " must not contain null bytes");
  }
  if (const auto it = value.find("enabled"); it != value.end()) {
    if (!it->is_boolean()) {
      return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                             sourceLabel(source, path + ".enabled") +
                                 " must be a boolean");
    }
    patch.enabled = it->get<bool>();
  }
  if (const auto it = value.find("session"); it != value.end()) {
    auto session = parseSessionPatch(*it, source, path + ".session", base);
    if (llvm::failed(session)) {
      return llvm::failure();
    }
    patch.session = (*std::move(session));
  }
  return patch;
}

[[nodiscard]] auto parseConfiguration(const Json& root,
                                      const std::filesystem::path& source,
                                      const std::filesystem::path& base)
    -> llvm::FailureOr<std::vector<DefinitionPatch>> {
  if (llvm::failed(requireObject(root, source, "$"))) {
    return llvm::failure();
  }
  if (llvm::failed(
          rejectUnknownKeys(root, {"schema-version", "qdmi"}, source, "$"))) {
    return llvm::failure();
  }
  const auto version = root.find("schema-version");
  if (version == root.end() || !version->is_number_integer() || *version != 1) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, "$.schema-version") +
                               " must be the integer 1");
  }
  const auto qdmiConfig = root.find("qdmi");
  if (qdmiConfig == root.end()) {
    return std::vector<DefinitionPatch>{};
  }
  if (llvm::failed(requireObject(*qdmiConfig, source, "$.qdmi"))) {
    return llvm::failure();
  }
  if (llvm::failed(
          rejectUnknownKeys(*qdmiConfig, {"devices"}, source, "$.qdmi"))) {
    return llvm::failure();
  }
  const auto devices = qdmiConfig->find("devices");
  if (devices == qdmiConfig->end()) {
    return std::vector<DefinitionPatch>{};
  }
  if (!devices->is_array()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, "$.qdmi.devices") +
                               " must be an array");
  }
  std::set<std::string> ids;
  std::vector<DefinitionPatch> patches;
  patches.reserve(devices->size());
  for (size_t i = 0; i < devices->size(); ++i) {
    auto result =
        parseDevicePatch((*devices)[i], source,
                         "$.qdmi.devices[" + std::to_string(i) + "]", base);
    if (llvm::failed(result)) {
      return llvm::failure();
    }
    auto& patch = (*result);
    if (!ids.emplace(patch.id).second) {
      return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                             sourceLabel(source, "$.qdmi.devices") +
                                 " contains duplicate id '" + patch.id + "'");
    }
    patches.emplace_back(std::move(patch));
  }
  return patches;
}

[[nodiscard]] llvm::FailureOr<Json>
parseJson(std::string_view text, const std::filesystem::path& source) {
  return mqt::detail::parseJSON(text, pathToString(source), nullptr,
                                QDMI_ERROR_INVALIDARGUMENT);
}

[[nodiscard]] llvm::FailureOr<Json>
readJson(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    return qdmi::emitError(QDMI_ERROR_NOTFOUND,
                           "Cannot open QDMI configuration file: " +
                               pathToString(path));
  }
  const std::string text{std::istreambuf_iterator<char>(stream), {}};
  if (stream.bad()) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           "Cannot read QDMI configuration file: " +
                               pathToString(path));
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

llvm::LogicalResult appendFragments(std::vector<std::filesystem::path>& files,
                                    const std::filesystem::path& directory) {
  const auto& absolute = directory;
  if (absolute.empty()) {
    return llvm::success();
  }
  std::error_code error;
  if (!std::filesystem::is_directory(absolute, error)) {
    return llvm::success();
  }
  std::vector<std::filesystem::path> found;
  for (std::filesystem::directory_iterator entry(absolute, error), end;
       !error && entry != end; entry.increment(error)) {
    const auto regular = entry->is_regular_file(error);
    if (error) {
      return qdmi::emitError(QDMI_ERROR_FATAL, pathToString(entry->path()) +
                                                   ": " + error.message());
    }
    if (regular && entry->path().filename().native().ends_with(
                       std::filesystem::path{".qdmi.json"}.native())) {
      found.emplace_back(entry->path());
    }
  }
  if (error) {
    return qdmi::emitError(QDMI_ERROR_FATAL,
                           pathToString(absolute) + ": " + error.message());
  }
  std::ranges::sort(found);
  files.insert(files.end(), found.begin(), found.end());
  return llvm::success();
}

[[nodiscard]] auto nearestProjectConfiguration(std::filesystem::path directory)
    -> llvm::FailureOr<std::optional<std::filesystem::path>> {
  while (!directory.empty()) {
    auto dedicated = directory / "qdmi.json";
    std::error_code error;
    if (std::filesystem::is_regular_file(dedicated, error)) {
      return std::optional<std::filesystem::path>{std::move(dedicated)};
    }
    if (error && error != std::errc::no_such_file_or_directory &&
        error != std::errc::not_a_directory) {
      return qdmi::emitError(QDMI_ERROR_FATAL,
                             pathToString(dedicated) + ": " + error.message());
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
    -> llvm::FailureOr<std::vector<std::filesystem::path>> {
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
    if (llvm::failed(appendFragments(files, directory))) {
      return llvm::failure();
    }
  }

  std::optional<std::filesystem::path> explicitFile;
  if (auto value = llvm::sys::Process::GetEnv("MQT_CORE_QDMI_CONFIG_FILE")) {
    explicitFile = pathFromString(*value);
  }
  if (explicitFile) {
    const auto resolved = resolvePath(*explicitFile, cwd);
    std::error_code error;
    if (!std::filesystem::is_regular_file(resolved, error)) {
      return qdmi::emitError(QDMI_ERROR_FATAL,
                             "Explicit QDMI configuration file does not "
                             "exist: " +
                                 pathToString(resolved));
    }
    files.emplace_back(resolved);
    return files;
  }

#ifdef _WIN32
  if (auto programData = llvm::sys::Process::GetEnv("PROGRAMDATA")) {
    appendIfFile(files,
                 pathFromString(*programData) / "mqt-core" / "qdmi.json");
  }
  if (auto appData = llvm::sys::Process::GetEnv("APPDATA")) {
    appendIfFile(files, pathFromString(*appData) / "mqt-core" / "qdmi.json");
  }
#else
  appendIfFile(files, "/etc/mqt-core/qdmi.json");
  if (auto xdg = llvm::sys::Process::GetEnv("XDG_CONFIG_HOME");
      xdg && !xdg->empty()) {
    appendIfFile(files, pathFromString(*xdg) / "mqt-core" / "qdmi.json");
  } else if (auto home = llvm::sys::Process::GetEnv("HOME")) {
    appendIfFile(files,
                 pathFromString(*home) / ".config" / "mqt-core" / "qdmi.json");
  }
#endif
  auto project = nearestProjectConfiguration(cwd);
  if (llvm::failed(project)) {
    return llvm::failure();
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
    -> llvm::FailureOr<qdmi::DeviceDefinition> {
  if (!patch.library || patch.library->empty()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           pathToString(patch.source) + ": enabled device '" +
                               patch.id + "' is missing library");
  }
  if (!patch.prefix || patch.prefix->empty()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           pathToString(patch.source) + ": enabled device '" +
                               patch.id + "' is missing prefix");
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
  try {
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
    if (!std::filesystem::is_regular_file(canonical, error) || error) {
      return QDMI_ERROR_LIBNOTFOUND;
    }

    auto root = readJson(canonical);
    if (llvm::failed(root)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    const auto patches =
        parseConfiguration(*root, canonical, canonical.parent_path());
    if (llvm::failed(patches)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    for (const auto& patch : *patches) {
      if (!patch.enabled.value_or(true)) {
        continue;
      }
      const auto definition = materialize(patch);
      if (llvm::failed(definition)) {
        return QDMI_ERROR_INVALIDARGUMENT;
      }
      if (!std::filesystem::is_regular_file(definition->library)) {
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
    auto paths = state.paths;
    auto storedIds = state.ids;
    paths.emplace_back(canonical);
    for (const auto& patch : *patches) {
      storedIds.emplace(patch.id);
    }
    state.paths.swap(paths);
    state.ids.swap(storedIds);
    return QDMI_SUCCESS;
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  } catch (const std::invalid_argument&) {
    return QDMI_ERROR_INVALIDARGUMENT;
  } catch (...) {
    return QDMI_ERROR_FATAL;
  }
}

auto parseDeviceSessionJson(const char* const data, const size_t size,
                            DeviceSessionConfig& config) -> int {
  config = {};
  if (data == nullptr && size == 0) {
    return QDMI_SUCCESS;
  }
  try {
    const auto value =
        parseJson(std::string_view{data, size}, "<device-session-json>");
    if (llvm::failed(value)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto parsed = parseSessionPatch(*value, "<device-session-json>", "$",
                                    std::filesystem::current_path());
    if (llvm::failed(parsed)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    config = std::move(*parsed);
    return QDMI_SUCCESS;
  } catch (const std::bad_alloc&) {
    return QDMI_ERROR_OUTOFMEM;
  } catch (const Json::parse_error&) {
    return QDMI_ERROR_INVALIDARGUMENT;
  } catch (const std::invalid_argument&) {
    return QDMI_ERROR_INVALIDARGUMENT;
  } catch (...) {
    return QDMI_ERROR_FATAL;
  }
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

llvm::FailureOr<DeviceRegistry> DeviceRegistry::discover() {
  std::error_code filesystemError;
  const auto cwd = std::filesystem::current_path(filesystemError);
  if (filesystemError) {
    return qdmi::emitError(QDMI_ERROR_FATAL, "Cannot read current directory: " +
                                                 filesystemError.message());
  }
  auto files = discoverFiles(cwd);
  if (llvm::failed(files)) {
    return llvm::failure();
  }
  std::map<std::string, DefinitionPatch> merged;
  const auto mergePatches =
      [&merged](const Json& root, const std::filesystem::path& source,
                const std::filesystem::path& base) -> llvm::LogicalResult {
    auto result = parseConfiguration(root, source, base);
    if (llvm::failed(result)) {
      return llvm::failure();
    }
    for (auto& patch : (*result)) {
      if (auto const it = merged.find(patch.id); it != merged.end()) {
        mergePatch(it->second, patch);
      } else {
        merged.emplace(patch.id, std::move(patch));
      }
    }
    return llvm::success();
  };
  auto staged = freezeDeviceManifests();
  files->insert(files->begin(), staged.begin(), staged.end());
  for (const auto& file : (*files)) {
    auto root = readJson(file);
    if (llvm::failed(root)) {
      return llvm::failure();
    }
    if (llvm::failed(mergePatches((*root), file, file.parent_path()))) {
      return llvm::failure();
    }
  }
  if (auto inlineJson =
          llvm::sys::Process::GetEnv("MQT_CORE_QDMI_CONFIG_JSON")) {
    auto root = parseJson(*inlineJson, "<MQT_CORE_QDMI_CONFIG_JSON>");
    if (llvm::failed(root)) {
      return llvm::failure();
    }
    if (llvm::failed(
            mergePatches((*root), "<MQT_CORE_QDMI_CONFIG_JSON>", cwd))) {
      return llvm::failure();
    }
  }
  DeviceRegistry registry;
  for (auto& [unused, patch] : merged) {
    if (!patch.enabled.value_or(true)) {
      registry.disabledIds_.emplace_back(std::move(patch.id));
    } else {
      auto definition = materialize(patch);
      if (llvm::failed(definition)) {
        return llvm::failure();
      }
      registry.definitions_.emplace_back((*std::move(definition)));
    }
  }
  return registry;
}

} // namespace qdmi::detail
