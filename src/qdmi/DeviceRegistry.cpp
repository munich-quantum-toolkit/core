/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/DeviceRegistry.hpp"

#include "DefaultDeviceRegistry.hpp"
#include "qdmi/SessionConfig.hpp"

#include <nlohmann/json.hpp> // NOLINT(misc-include-cleaner)

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>

#include <memory>
#else
#include <dlfcn.h>
#endif

namespace qdmi {
namespace {
using Json = nlohmann::json; // NOLINT(misc-include-cleaner)

struct SessionPatch {
  std::optional<std::string> baseUrl;
  std::optional<std::string> token;
  std::optional<std::filesystem::path> authFile;
  std::optional<std::string> authUrl;
  std::optional<std::string> username;
  std::optional<std::string> password;
  std::optional<DeviceConfigurationSource> deviceConfiguration;
  std::optional<std::string> custom1;
  std::optional<std::string> custom2;
  std::optional<std::string> custom3;
  std::optional<std::string> custom4;
  std::optional<std::string> custom5;
};

struct DefinitionPatch {
  std::string id;
  std::optional<std::filesystem::path> library;
  std::optional<std::string> prefix;
  std::optional<bool> enabled;
  SessionPatch session;
  std::filesystem::path source;
};

[[nodiscard]] std::string sourceLabel(const std::filesystem::path& source,
                                      const std::string_view path) {
  return source.string() + ":" + std::string(path);
}

void requireObject(const Json& value, const std::filesystem::path& source,
                   const std::string_view path) {
  if (!value.is_object()) {
    throw std::invalid_argument(sourceLabel(source, path) +
                                " must be an object");
  }
}

void rejectUnknownKeys(const Json& value,
                       const std::initializer_list<std::string_view> allowed,
                       const std::filesystem::path& source,
                       const std::string_view path) {
  const std::set<std::string_view> known(allowed);
  for (const auto& [key, unused] : value.items()) {
    static_cast<void>(unused);
    if (!known.contains(key)) {
      throw std::invalid_argument(sourceLabel(source, path) +
                                  " contains unknown key '" + key + "'");
    }
  }
}

[[nodiscard]] std::optional<std::string>
optionalString(const Json& value, const std::string& key,
               const std::filesystem::path& source, const std::string& path) {
  const auto it = value.find(key);
  if (it == value.end()) {
    return std::nullopt;
  }
  if (!it->is_string()) {
    throw std::invalid_argument(sourceLabel(source, path + "." + key) +
                                " must be a string");
  }
  return it->get<std::string>();
}

[[nodiscard]] std::filesystem::path
resolvePath(std::filesystem::path path, const std::filesystem::path& base) {
  if (path.is_relative()) {
    path = base / path;
  }
  return path.lexically_normal();
}

[[nodiscard]] std::filesystem::path
absolutePath(const std::filesystem::path& path) {
  if (path.empty()) {
    return {};
  }
  return std::filesystem::absolute(path).lexically_normal();
}

[[nodiscard]] SessionPatch
parseSessionPatch(const Json& value, const std::filesystem::path& source,
                  const std::string& path, const std::filesystem::path& base) {
  requireObject(value, source, path);
  rejectUnknownKeys(value,
                    {"base-url", "token", "auth-file", "auth-url", "username",
                     "password", "custom1", "custom2", "custom3", "custom4",
                     "custom5", "device-config"},
                    source, path);
  SessionPatch patch;
  patch.baseUrl = optionalString(value, "base-url", source, path);
  patch.token = optionalString(value, "token", source, path);
  patch.authUrl = optionalString(value, "auth-url", source, path);
  patch.username = optionalString(value, "username", source, path);
  patch.password = optionalString(value, "password", source, path);
  patch.custom1 = optionalString(value, "custom1", source, path);
  patch.custom2 = optionalString(value, "custom2", source, path);
  patch.custom3 = optionalString(value, "custom3", source, path);
  patch.custom4 = optionalString(value, "custom4", source, path);
  patch.custom5 = optionalString(value, "custom5", source, path);
  if (const auto config = value.find("device-config"); config != value.end()) {
    const auto configPath = path + ".device-config";
    requireObject(*config, source, configPath);
    rejectUnknownKeys(*config, {"inline", "file"}, source, configPath);
    const auto inlineConfig = config->find("inline");
    const auto fileConfig = config->find("file");
    if ((inlineConfig == config->end()) == (fileConfig == config->end())) {
      throw std::invalid_argument(sourceLabel(source, configPath) +
                                  " must contain exactly one of 'inline' and "
                                  "'file'");
    }
    if (inlineConfig != config->end()) {
      if (!inlineConfig->is_object()) {
        throw std::invalid_argument(
            sourceLabel(source, configPath + ".inline") + " must be an object");
      }
      patch.deviceConfiguration =
          InlineDeviceConfiguration{.json = inlineConfig->dump()};
    } else {
      if (!fileConfig->is_string() ||
          fileConfig->get_ref<const std::string&>().empty()) {
        throw std::invalid_argument(sourceLabel(source, configPath + ".file") +
                                    " must be a non-empty string");
      }
      patch.deviceConfiguration = FileDeviceConfiguration{
          .path = resolvePath(fileConfig->get<std::string>(), base)};
    }
  }
  if (auto authFile = optionalString(value, "auth-file", source, path)) {
    patch.authFile = resolvePath(*authFile, base);
  }
  return patch;
}

[[nodiscard]] DefinitionPatch
parseDevicePatch(const Json& value, const std::filesystem::path& source,
                 const std::string& path, const std::filesystem::path& base) {
  requireObject(value, source, path);
  rejectUnknownKeys(value, {"id", "library", "prefix", "enabled", "session"},
                    source, path);
  const auto id = optionalString(value, "id", source, path);
  if (!id || id->empty()) {
    throw std::invalid_argument(sourceLabel(source, path + ".id") +
                                " must be a non-empty string");
  }
  DefinitionPatch patch;
  patch.id = *id;
  patch.source = source;
  if (auto library = optionalString(value, "library", source, path)) {
    patch.library = resolvePath(*library, base);
  }
  patch.prefix = optionalString(value, "prefix", source, path);
  if (const auto it = value.find("enabled"); it != value.end()) {
    if (!it->is_boolean()) {
      throw std::invalid_argument(sourceLabel(source, path + ".enabled") +
                                  " must be a boolean");
    }
    patch.enabled = it->get<bool>();
  }
  if (const auto it = value.find("session"); it != value.end()) {
    patch.session = parseSessionPatch(*it, source, path + ".session", base);
  }
  return patch;
}

[[nodiscard]] std::vector<DefinitionPatch>
parseConfiguration(const Json& root, const std::filesystem::path& source,
                   const std::filesystem::path& base) {
  requireObject(root, source, "$");
  rejectUnknownKeys(root, {"schema-version", "qdmi"}, source, "$");
  const auto version = root.find("schema-version");
  if (version == root.end() || !version->is_number_integer() ||
      version->get<int>() != 1) {
    throw std::invalid_argument(sourceLabel(source, "$.schema-version") +
                                " must be the integer 1");
  }
  const auto qdmiConfig = root.find("qdmi");
  if (qdmiConfig == root.end()) {
    return {};
  }
  requireObject(*qdmiConfig, source, "$.qdmi");
  rejectUnknownKeys(*qdmiConfig, {"devices"}, source, "$.qdmi");
  const auto devices = qdmiConfig->find("devices");
  if (devices == qdmiConfig->end()) {
    return {};
  }
  if (!devices->is_array()) {
    throw std::invalid_argument(sourceLabel(source, "$.qdmi.devices") +
                                " must be an array");
  }
  std::set<std::string> ids;
  std::vector<DefinitionPatch> patches;
  patches.reserve(devices->size());
  for (size_t i = 0; i < devices->size(); ++i) {
    auto patch =
        parseDevicePatch((*devices)[i], source,
                         "$.qdmi.devices[" + std::to_string(i) + "]", base);
    if (!ids.emplace(patch.id).second) {
      throw std::invalid_argument(sourceLabel(source, "$.qdmi.devices") +
                                  " contains duplicate id '" + patch.id + "'");
    }
    patches.emplace_back(std::move(patch));
  }
  return patches;
}

[[nodiscard]] Json readJson(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("Cannot open QDMI configuration file: " +
                             path.string());
  }
  try {
    return Json::parse(stream);
  } catch (const Json::parse_error& error) {
    throw std::invalid_argument(path.string() +
                                ": invalid JSON: " + error.what());
  }
}

template <class T>
void mergeOptional(std::optional<T>& target, const std::optional<T>& source) {
  if (source) {
    target = source;
  }
}

void mergeSession(SessionPatch& target, const SessionPatch& source) {
  mergeOptional(target.baseUrl, source.baseUrl);
  mergeOptional(target.token, source.token);
  mergeOptional(target.authFile, source.authFile);
  mergeOptional(target.authUrl, source.authUrl);
  mergeOptional(target.username, source.username);
  mergeOptional(target.password, source.password);
  mergeOptional(target.deviceConfiguration, source.deviceConfiguration);
  mergeOptional(target.custom1, source.custom1);
  mergeOptional(target.custom2, source.custom2);
  mergeOptional(target.custom3, source.custom3);
  mergeOptional(target.custom4, source.custom4);
  mergeOptional(target.custom5, source.custom5);
}

void mergePatch(DefinitionPatch& target, const DefinitionPatch& source) {
  mergeOptional(target.library, source.library);
  mergeOptional(target.prefix, source.prefix);
  mergeOptional(target.enabled, source.enabled);
  mergeSession(target.session, source.session);
  target.source = source.source;
}

[[nodiscard]] std::filesystem::path moduleDirectory() {
#ifdef _WIN32
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&moduleDirectory),
                         &module) == 0) {
    return {};
  }
  std::wstring buffer(MAX_PATH, L'\0');
  while (true) {
    const auto size = GetModuleFileNameW(module, buffer.data(),
                                         static_cast<DWORD>(buffer.size()));
    if (size == 0) {
      return {};
    }
    if (size < buffer.size()) {
      buffer.resize(size);
      return std::filesystem::path(buffer).parent_path();
    }
    buffer.resize(buffer.size() * 2);
  }
#else
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&moduleDirectory), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

[[nodiscard]] std::optional<std::string> environment(const char* name) {
#ifdef _WIN32
  char* raw = nullptr;
  size_t size = 0;
  if (_dupenv_s(&raw, &size, name) != 0 || raw == nullptr) {
    return std::nullopt;
  }
  const std::unique_ptr<char, decltype(&std::free)> value(raw, &std::free);
  if (*value == '\0') {
    return std::nullopt;
  }
  return std::string(value.get());
#else
  if (const auto* value = std::getenv(name);
      value != nullptr && *value != '\0') {
    return std::string(value);
  }
  return std::nullopt;
#endif
}

void appendIfFile(std::vector<std::filesystem::path>& files,
                  const std::filesystem::path& path) {
  const auto absolute = absolutePath(path);
  if (absolute.empty()) {
    return;
  }
  std::error_code error;
  if (std::filesystem::is_regular_file(absolute, error)) {
    files.emplace_back(absolute);
  }
}

void appendFragments(std::vector<std::filesystem::path>& files,
                     const std::filesystem::path& directory) {
  const auto absolute = absolutePath(directory);
  if (absolute.empty()) {
    return;
  }
  std::error_code error;
  if (!std::filesystem::is_directory(absolute, error)) {
    return;
  }
  std::vector<std::filesystem::path> found;
  for (const auto& entry : std::filesystem::directory_iterator(absolute)) {
    if (entry.is_regular_file() &&
        entry.path().filename().string().ends_with(".qdmi.json")) {
      found.emplace_back(entry.path());
    }
  }
  std::ranges::sort(found);
  files.insert(files.end(), found.begin(), found.end());
}

[[nodiscard]] std::optional<std::filesystem::path>
nearestProjectConfiguration(std::filesystem::path directory) {
  while (!directory.empty()) {
    auto dedicated = directory / "qdmi.json";
    if (std::filesystem::is_regular_file(dedicated)) {
      return dedicated;
    }
    const auto parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return std::nullopt;
}

[[nodiscard]] std::vector<std::filesystem::path> discoverFiles() {
  std::vector<std::filesystem::path> files;
  const auto root = moduleDirectory();
  appendFragments(files, root);
  appendFragments(files, root / "bin");
  appendFragments(files, root / "lib");
  appendFragments(files, root / "mqt-core" / "qdmi");
  appendFragments(files, root / "qdmi");

  std::optional<std::filesystem::path> explicitFile;
  if (auto value = environment("MQT_CORE_QDMI_CONFIG_FILE")) {
    explicitFile = *value;
  }
  if (explicitFile) {
    const auto resolved =
        resolvePath(*explicitFile, std::filesystem::current_path());
    if (!std::filesystem::is_regular_file(resolved)) {
      throw std::runtime_error("Explicit QDMI configuration file does not "
                               "exist: " +
                               resolved.string());
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
  if (auto project =
          nearestProjectConfiguration(std::filesystem::current_path())) {
    files.emplace_back(std::move(*project));
  }
  return files;
}

[[nodiscard]] std::optional<DeviceDefinition>
materialize(const DefinitionPatch& patch) {
  if (!patch.enabled.value_or(true)) {
    return std::nullopt;
  }
  if (!patch.library || patch.library->empty()) {
    throw std::invalid_argument(patch.source.string() + ": enabled device '" +
                                patch.id + "' is missing library");
  }
  if (!patch.prefix || patch.prefix->empty()) {
    throw std::invalid_argument(patch.source.string() + ": enabled device '" +
                                patch.id + "' is missing prefix");
  }
  DeviceDefinition definition;
  definition.id = patch.id;
  definition.library = *patch.library;
  definition.prefix = *patch.prefix;
  definition.session.baseUrl = patch.session.baseUrl;
  definition.session.token = patch.session.token;
  if (patch.session.authFile) {
    definition.session.authFile = patch.session.authFile;
  }
  definition.session.authUrl = patch.session.authUrl;
  definition.session.username = patch.session.username;
  definition.session.password = patch.session.password;
  definition.session.deviceConfiguration = patch.session.deviceConfiguration;
  definition.session.custom1 = patch.session.custom1;
  definition.session.custom2 = patch.session.custom2;
  definition.session.custom3 = patch.session.custom3;
  definition.session.custom4 = patch.session.custom4;
  definition.session.custom5 = patch.session.custom5;
  return definition;
}

} // namespace

DeviceRegistry::DeviceRegistry() {
  std::map<std::string, DefinitionPatch> merged;
  const auto mergePatches = [&merged](std::vector<DefinitionPatch> patches) {
    for (auto& patch : patches) {
      if (auto it = merged.find(patch.id); it != merged.end()) {
        mergePatch(it->second, patch);
      } else {
        merged.emplace(patch.id, std::move(patch));
      }
    }
  };

  for (const auto& file : discoverFiles()) {
    mergePatches(parseConfiguration(readJson(file), file, file.parent_path()));
  }
  const auto inlineBase = std::filesystem::current_path();
  if (auto inlineJson = environment("MQT_CORE_QDMI_CONFIG_JSON")) {
    try {
      mergePatches(parseConfiguration(
          Json::parse(*inlineJson), "<MQT_CORE_QDMI_CONFIG_JSON>", inlineBase));
    } catch (const Json::parse_error& error) {
      throw std::invalid_argument(
          std::string("<MQT_CORE_QDMI_CONFIG_JSON>: invalid JSON: ") +
          error.what());
    }
  }
  for (auto& [unused, patch] : merged) {
    static_cast<void>(unused);
    if (!patch.enabled.value_or(true)) {
      disabledIds_.emplace(std::move(patch.id));
    } else if (auto definition = materialize(patch)) {
      definitions_.emplace_back(std::move(*definition));
    }
  }
}

DeviceRegistry::DeviceRegistry(std::vector<DeviceDefinition> definitions) {
  for (auto& definition : definitions) {
    registerDevice(std::move(definition));
  }
}

std::vector<std::string> DeviceRegistry::deviceIds() const {
  std::vector<std::string> ids;
  ids.reserve(definitions_.size());
  std::ranges::transform(definitions_, std::back_inserter(ids),
                         &DeviceDefinition::id);
  return ids;
}

namespace {

void validateDefinition(const DeviceDefinition& definition) {
  if (definition.id.empty()) {
    throw std::invalid_argument("Device definition ID must not be empty");
  }
  if (definition.library.empty()) {
    throw std::invalid_argument("Device definition library must not be empty");
  }
  if (definition.prefix.empty()) {
    throw std::invalid_argument("Device definition prefix must not be empty");
  }
}

struct DefaultRegistryState {
  std::mutex mutex;
  DeviceRegistry registry;
};

DefaultRegistryState& defaultRegistryState() {
  static DefaultRegistryState state;
  return state;
}

} // namespace

void DeviceRegistry::registerDevice(DeviceDefinition definition,
                                    const bool replace) {
  validateDefinition(definition);
  if (disabledIds_.contains(definition.id)) {
    if (!replace) {
      throw std::invalid_argument("QDMI device ID '" + definition.id +
                                  "' is disabled by configuration");
    }
    disabledIds_.erase(definition.id);
  }

  const auto existing =
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id);
  if (existing == definitions_.end()) {
    definitions_.emplace_back(std::move(definition));
    return;
  }
  if (!replace) {
    throw std::invalid_argument("QDMI device ID '" + definition.id +
                                "' is already registered");
  }
  *existing = std::move(definition);
}

bool DeviceRegistry::registerDeviceIfAbsent(DeviceDefinition definition) {
  validateDefinition(definition);
  if (disabledIds_.contains(definition.id) ||
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id) !=
          definitions_.end()) {
    return false;
  }
  definitions_.emplace_back(std::move(definition));
  return true;
}

void registerDevice(DeviceDefinition definition, const bool replace) {
  auto& state = defaultRegistryState();
  const std::scoped_lock lock(state.mutex);
  state.registry.registerDevice(std::move(definition), replace);
}

bool registerDeviceIfAbsent(DeviceDefinition definition) {
  auto& state = defaultRegistryState();
  const std::scoped_lock lock(state.mutex);
  return state.registry.registerDeviceIfAbsent(std::move(definition));
}

std::vector<std::string> registeredDeviceIds() {
  auto& state = defaultRegistryState();
  const std::scoped_lock lock(state.mutex);
  return state.registry.deviceIds();
}

DeviceRegistry detail::snapshotDefaultDeviceRegistry() {
  auto& state = defaultRegistryState();
  const std::scoped_lock lock(state.mutex);
  return state.registry;
}

} // namespace qdmi
