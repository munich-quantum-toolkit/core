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

#include "mlir/Support/LogicalResult.h"

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
mlir::LogicalResult validateDeviceId(const std::string_view id) {
  if (id.empty()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           "Device definition ID must not be empty");
  }
  if (id.find('\0') != std::string_view::npos) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           "Device definition ID must not contain NUL");
  }
  return mlir::success();
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

mlir::LogicalResult requireObject(const Json& value,
                                  const std::filesystem::path& source,
                                  const std::string_view path) {
  if (!value.is_object()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path) + " must be an object");
  }
  return mlir::success();
}

mlir::LogicalResult rejectUnknownKeys(
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
  return mlir::success();
}

[[nodiscard]] auto optionalString(const Json& value, const std::string& key,
                                  const std::filesystem::path& source,
                                  const std::string& path)
    -> mlir::FailureOr<std::optional<std::string>> {
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
    -> mlir::FailureOr<DeviceSessionConfig> {
  if (mlir::failed(requireObject(value, source, path))) {
    return mlir::failure();
  }
  if (mlir::failed(rejectUnknownKeys(value,
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
    return mlir::failure();
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
    if (mlir::failed(result)) {
      return mlir::failure();
    }
    *destination = (*std::move(result));
  }
  if (const auto config = value.find("device-config"); config != value.end()) {
    const auto configPath = path + ".device-config";
    if (mlir::failed(requireObject(*config, source, configPath))) {
      return mlir::failure();
    }
    if (mlir::failed(rejectUnknownKeys(*config, {"inline", "file"}, source,
                                       configPath))) {
      return mlir::failure();
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
  if (mlir::failed(authFile)) {
    return mlir::failure();
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
    -> mlir::FailureOr<DefinitionPatch> {
  if (mlir::failed(requireObject(value, source, path))) {
    return mlir::failure();
  }
  if (mlir::failed(rejectUnknownKeys(
          value, {"id", "library", "prefix", "enabled", "session"}, source,
          path))) {
    return mlir::failure();
  }
  auto idResult = optionalString(value, "id", source, path);
  if (mlir::failed(idResult)) {
    return mlir::failure();
  }
  const auto& id = (*idResult);
  if (!id || id->empty()) {
    return qdmi::emitError(QDMI_ERROR_INVALIDARGUMENT,
                           sourceLabel(source, path + ".id") +
                               " must be a non-empty string");
  }
  if (mlir::failed(validateDeviceId(*id))) {
    return mlir::failure();
  }
  DefinitionPatch patch;
  patch.id = *id;
  patch.source = source;
  auto library = optionalString(value, "library", source, path);
  if (mlir::failed(library)) {
    return mlir::failure();
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
  if (mlir::failed(prefix)) {
    return mlir::failure();
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
    if (mlir::failed(session)) {
      return mlir::failure();
    }
    patch.session = (*std::move(session));
  }
  return patch;
}

[[nodiscard]] auto parseConfiguration(const Json& root,
                                      const std::filesystem::path& source,
                                      const std::filesystem::path& base)
    -> mlir::FailureOr<std::vector<DefinitionPatch>> {
  if (mlir::failed(requireObject(root, source, "$"))) {
    return mlir::failure();
  }
  if (mlir::failed(
          rejectUnknownKeys(root, {"schema-version", "qdmi"}, source, "$"))) {
    return mlir::failure();
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
  if (mlir::failed(requireObject(*qdmiConfig, source, "$.qdmi"))) {
    return mlir::failure();
  }
  if (mlir::failed(
          rejectUnknownKeys(*qdmiConfig, {"devices"}, source, "$.qdmi"))) {
    return mlir::failure();
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
    if (mlir::failed(result)) {
      return mlir::failure();
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

[[nodiscard]] mlir::FailureOr<Json>
parseJson(std::string_view text, const std::filesystem::path& source) {
  return mqt::detail::parseJSON(text, pathToString(source), nullptr,
                                QDMI_ERROR_INVALIDARGUMENT);
}

[[nodiscard]] mlir::FailureOr<Json>
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

mlir::LogicalResult appendFragments(std::vector<std::filesystem::path>& files,
                                    const std::filesystem::path& directory) {
  const auto& absolute = directory;
  if (absolute.empty()) {
    return mlir::success();
  }
  std::error_code error;
  if (!std::filesystem::is_directory(absolute, error)) {
    return mlir::success();
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
  return mlir::success();
}

[[nodiscard]] auto nearestProjectConfiguration(std::filesystem::path directory)
    -> mlir::FailureOr<std::optional<std::filesystem::path>> {
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
    -> mlir::FailureOr<std::vector<std::filesystem::path>> {
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
    if (mlir::failed(appendFragments(files, directory))) {
      return mlir::failure();
    }
  }

  std::optional<std::filesystem::path> explicitFile;
  if (auto value = environment("MQT_CORE_QDMI_CONFIG_FILE")) {
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
  if (auto programData = environment("PROGRAMDATA")) {
    appendIfFile(files,
                 pathFromString(*programData) / "mqt-core" / "qdmi.json");
  }
  if (auto appData = environment("APPDATA")) {
    appendIfFile(files, pathFromString(*appData) / "mqt-core" / "qdmi.json");
  }
#else
  appendIfFile(files, "/etc/mqt-core/qdmi.json");
  if (auto xdg = environment("XDG_CONFIG_HOME")) {
    appendIfFile(files, pathFromString(*xdg) / "mqt-core" / "qdmi.json");
  } else if (auto home = environment("HOME")) {
    appendIfFile(files,
                 pathFromString(*home) / ".config" / "mqt-core" / "qdmi.json");
  }
#endif
  auto project = nearestProjectConfiguration(cwd);
  if (mlir::failed(project)) {
    return mlir::failure();
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
    -> mlir::FailureOr<qdmi::DeviceDefinition> {
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
    if (mlir::failed(root)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    const auto patches =
        parseConfiguration(*root, canonical, canonical.parent_path());
    if (mlir::failed(patches)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    for (const auto& patch : *patches) {
      if (!patch.enabled.value_or(true)) {
        continue;
      }
      const auto definition = materialize(patch);
      if (mlir::failed(definition)) {
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
    if (mlir::failed(value)) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    auto parsed = parseSessionPatch(*value, "<device-session-json>", "$",
                                    std::filesystem::current_path());
    if (mlir::failed(parsed)) {
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

mlir::FailureOr<DeviceRegistry> DeviceRegistry::discover() {
  std::error_code filesystemError;
  const auto cwd = std::filesystem::current_path(filesystemError);
  if (filesystemError) {
    return qdmi::emitError(QDMI_ERROR_FATAL, "Cannot read current directory: " +
                                                 filesystemError.message());
  }
  auto files = discoverFiles(cwd);
  if (mlir::failed(files)) {
    return mlir::failure();
  }
  std::map<std::string, DefinitionPatch> merged;
  const auto mergePatches =
      [&merged](const Json& root, const std::filesystem::path& source,
                const std::filesystem::path& base) -> mlir::LogicalResult {
    auto result = parseConfiguration(root, source, base);
    if (mlir::failed(result)) {
      return mlir::failure();
    }
    for (auto& patch : (*result)) {
      if (auto const it = merged.find(patch.id); it != merged.end()) {
        mergePatch(it->second, patch);
      } else {
        merged.emplace(patch.id, std::move(patch));
      }
    }
    return mlir::success();
  };
  auto staged = freezeDeviceManifests();
  files->insert(files->begin(), staged.begin(), staged.end());
  for (const auto& file : (*files)) {
    auto root = readJson(file);
    if (mlir::failed(root)) {
      return mlir::failure();
    }
    if (mlir::failed(mergePatches((*root), file, file.parent_path()))) {
      return mlir::failure();
    }
  }
  if (auto inlineJson = environment("MQT_CORE_QDMI_CONFIG_JSON")) {
    auto root = parseJson(*inlineJson, "<MQT_CORE_QDMI_CONFIG_JSON>");
    if (mlir::failed(root)) {
      return mlir::failure();
    }
    if (mlir::failed(
            mergePatches((*root), "<MQT_CORE_QDMI_CONFIG_JSON>", cwd))) {
      return mlir::failure();
    }
  }
  DeviceRegistry registry;
  for (auto& [unused, patch] : merged) {
    if (!patch.enabled.value_or(true)) {
      registry.disabledIds_.emplace_back(std::move(patch.id));
    } else {
      auto definition = materialize(patch);
      if (mlir::failed(definition)) {
        return mlir::failure();
      }
      registry.definitions_.emplace_back((*std::move(definition)));
    }
  }
  return registry;
}

} // namespace qdmi::detail
