/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qdmi/DeviceManager.hpp"

#include "qdmi/driver/Driver.hpp"

#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace qdmi {
namespace {
using Json = nlohmann::json;

struct SessionPatch {
  std::optional<std::string> baseUrl;
  std::optional<std::string> token;
  std::optional<std::filesystem::path> authFile;
  std::optional<std::string> authUrl;
  std::optional<std::string> username;
  std::optional<std::string> password;
  std::optional<std::string> projectId;
  std::optional<std::string> custom1;
  std::optional<std::string> custom2;
  std::optional<std::string> custom3;
  std::optional<std::string> custom4;
  std::optional<std::string> custom5;
};

struct DefinitionPatch {
  std::string id;
  std::optional<std::filesystem::path> library;
  std::optional<std::string> abi;
  std::optional<std::string> prefix;
  std::optional<bool> enabled;
  SessionPatch session;
  std::filesystem::path source;
};

[[nodiscard]] auto sourceLabel(const std::filesystem::path& source,
                               const std::string_view path) -> std::string {
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

[[nodiscard]] auto optionalString(const Json& value, const std::string& key,
                                  const std::filesystem::path& source,
                                  const std::string& path)
    -> std::optional<std::string> {
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
    -> SessionPatch {
  requireObject(value, source, path);
  rejectUnknownKeys(value,
                    {"base-url", "token", "auth-file", "auth-url", "username",
                     "password", "project-id", "custom1", "custom2", "custom3",
                     "custom4", "custom5"},
                    source, path);
  SessionPatch patch;
  patch.baseUrl = optionalString(value, "base-url", source, path);
  patch.token = optionalString(value, "token", source, path);
  patch.authUrl = optionalString(value, "auth-url", source, path);
  patch.username = optionalString(value, "username", source, path);
  patch.password = optionalString(value, "password", source, path);
  patch.projectId = optionalString(value, "project-id", source, path);
  patch.custom1 = optionalString(value, "custom1", source, path);
  patch.custom2 = optionalString(value, "custom2", source, path);
  patch.custom3 = optionalString(value, "custom3", source, path);
  patch.custom4 = optionalString(value, "custom4", source, path);
  patch.custom5 = optionalString(value, "custom5", source, path);
  if (auto authFile = optionalString(value, "auth-file", source, path)) {
    patch.authFile = resolvePath(*authFile, base);
  }
  return patch;
}

[[nodiscard]] auto
parseDevicePatch(const Json& value, const std::filesystem::path& source,
                 const std::string& path, const std::filesystem::path& base)
    -> DefinitionPatch {
  requireObject(value, source, path);
  rejectUnknownKeys(value,
                    {"id", "library", "abi", "prefix", "enabled", "session"},
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
  patch.abi = optionalString(value, "abi", source, path);
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

[[nodiscard]] auto parseConfiguration(const Json& root,
                                      const std::filesystem::path& source,
                                      const std::filesystem::path& base)
    -> std::vector<DefinitionPatch> {
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
  rejectUnknownKeys(*qdmiConfig, {"devices", "device-config"}, source,
                    "$.qdmi");
  if (const auto deviceConfig = qdmiConfig->find("device-config");
      deviceConfig != qdmiConfig->end()) {
    requireObject(*deviceConfig, source, "$.qdmi.device-config");
  }
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

[[nodiscard]] auto readJson(const std::filesystem::path& path) -> Json {
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

[[nodiscard]] auto readPyproject(const std::filesystem::path& path)
    -> std::optional<Json> {
  try {
    const auto table = toml::parse_file(path.string());
    const auto* qdmiTable = table["tool"]["mqt-core"]["qdmi"].as_table();
    if (qdmiTable == nullptr) {
      return std::nullopt;
    }
    std::ostringstream formatted;
    formatted << toml::json_formatter{*qdmiTable};
    return Json{{"schema-version", 1}, {"qdmi", Json::parse(formatted.str())}};
  } catch (const toml::parse_error& error) {
    throw std::invalid_argument(
        path.string() + ": invalid TOML: " + std::string(error.description()));
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
  mergeOptional(target.projectId, source.projectId);
  mergeOptional(target.custom1, source.custom1);
  mergeOptional(target.custom2, source.custom2);
  mergeOptional(target.custom3, source.custom3);
  mergeOptional(target.custom4, source.custom4);
  mergeOptional(target.custom5, source.custom5);
}

void mergePatch(DefinitionPatch& target, const DefinitionPatch& source) {
  mergeOptional(target.library, source.library);
  mergeOptional(target.abi, source.abi);
  mergeOptional(target.prefix, source.prefix);
  mergeOptional(target.enabled, source.enabled);
  mergeSession(target.session, source.session);
  target.source = source.source;
}

[[nodiscard]] auto moduleDirectory() -> std::filesystem::path {
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

[[nodiscard]] auto environment(const char* name) -> std::optional<std::string> {
  if (const auto* value = std::getenv(name);
      value != nullptr && *value != '\0') {
    return std::string(value);
  }
  return std::nullopt;
}

void appendIfFile(std::vector<std::filesystem::path>& files,
                  const std::filesystem::path& path) {
  std::error_code error;
  if (std::filesystem::is_regular_file(path, error)) {
    files.emplace_back(path);
  }
}

void appendFragments(std::vector<std::filesystem::path>& files,
                     const std::filesystem::path& directory) {
  std::error_code error;
  if (!std::filesystem::is_directory(directory, error)) {
    return;
  }
  std::vector<std::filesystem::path> found;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() &&
        entry.path().filename().string().ends_with(".qdmi.json")) {
      found.emplace_back(entry.path());
    }
  }
  std::ranges::sort(found);
  files.insert(files.end(), found.begin(), found.end());
}

[[nodiscard]] auto nearestProjectConfiguration(std::filesystem::path directory)
    -> std::optional<std::filesystem::path> {
  while (!directory.empty()) {
    const auto dedicated = directory / "mqt-core.json";
    if (std::filesystem::is_regular_file(dedicated)) {
      return dedicated;
    }
    const auto pyproject = directory / "pyproject.toml";
    if (std::filesystem::is_regular_file(pyproject)) {
      if (readPyproject(pyproject)) {
        return pyproject;
      }
    }
    const auto parent = directory.parent_path();
    if (parent == directory) {
      break;
    }
    directory = parent;
  }
  return std::nullopt;
}

[[nodiscard]] auto discoverFiles(const ConfigOptions& options)
    -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> files;
  if (!options.isolated) {
    const auto root = options.configRoot.value_or(moduleDirectory());
    appendFragments(files, root);
    appendFragments(files, root / "lib");
    appendFragments(files, root / "mqt-core" / "qdmi");
    appendFragments(files, root / "qdmi");
  }

  auto explicitFile = options.explicitFile;
  if (!explicitFile) {
    if (auto value = environment("MQT_CORE_QDMI_CONFIG_FILE")) {
      explicitFile = *value;
    }
  }
  if (explicitFile) {
    const auto resolved = resolvePath(
        *explicitFile,
        options.baseDirectory.value_or(std::filesystem::current_path()));
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
                            "mqt-core.json");
  }
  if (auto appData = environment("APPDATA")) {
    appendIfFile(files, std::filesystem::path(*appData) / "mqt-core" /
                            "mqt-core.json");
  }
#else
  appendIfFile(files, "/etc/mqt-core/mqt-core.json");
  if (auto xdg = environment("XDG_CONFIG_HOME")) {
    appendIfFile(files,
                 std::filesystem::path(*xdg) / "mqt-core" / "mqt-core.json");
  } else if (auto home = environment("HOME")) {
    appendIfFile(files, std::filesystem::path(*home) / ".config" / "mqt-core" /
                            "mqt-core.json");
  }
#endif
  if (auto project = nearestProjectConfiguration(
          options.baseDirectory.value_or(std::filesystem::current_path()))) {
    files.emplace_back(std::move(*project));
  }
  return files;
}

[[nodiscard]] auto toPatch(const DeviceDefinition& definition,
                           const std::filesystem::path& base)
    -> DefinitionPatch {
  DefinitionPatch patch;
  patch.id = definition.id;
  if (!definition.library.empty()) {
    patch.library = resolvePath(definition.library, base);
  }
  patch.abi = definition.abi;
  patch.prefix = definition.prefix;
  patch.enabled = definition.enabled;
  patch.source = definition.source.empty() ? std::filesystem::path("<runtime>")
                                           : definition.source;
  patch.session.baseUrl = definition.session.baseUrl;
  patch.session.token = definition.session.token;
  if (definition.session.authFile) {
    patch.session.authFile = resolvePath(*definition.session.authFile, base);
  }
  patch.session.authUrl = definition.session.authUrl;
  patch.session.username = definition.session.username;
  patch.session.password = definition.session.password;
  patch.session.projectId = definition.session.projectId;
  patch.session.custom1 = definition.session.custom1;
  patch.session.custom2 = definition.session.custom2;
  patch.session.custom3 = definition.session.custom3;
  patch.session.custom4 = definition.session.custom4;
  patch.session.custom5 = definition.session.custom5;
  return patch;
}

[[nodiscard]] auto materialize(const DefinitionPatch& patch)
    -> std::optional<DeviceDefinition> {
  if (!patch.enabled.value_or(true)) {
    return std::nullopt;
  }
  const auto abi = patch.abi.value_or("qdmi-v1");
  if (abi != "qdmi-v1") {
    throw std::invalid_argument(patch.source.string() + ": device '" +
                                patch.id + "' uses unsupported ABI '" + abi +
                                "'");
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
  definition.abi = abi;
  definition.prefix = *patch.prefix;
  definition.enabled = true;
  definition.source = patch.source;
  definition.session.baseUrl = patch.session.baseUrl;
  definition.session.token = patch.session.token;
  definition.session.authFile = patch.session.authFile;
  definition.session.authUrl = patch.session.authUrl;
  definition.session.username = patch.session.username;
  definition.session.password = patch.session.password;
  definition.session.projectId = patch.session.projectId;
  definition.session.custom1 = patch.session.custom1;
  definition.session.custom2 = patch.session.custom2;
  definition.session.custom3 = patch.session.custom3;
  definition.session.custom4 = patch.session.custom4;
  definition.session.custom5 = patch.session.custom5;
  return definition;
}

void overlay(SessionParameters& target, const SessionParameters& source) {
  mergeOptional(target.baseUrl, source.baseUrl);
  mergeOptional(target.token, source.token);
  mergeOptional(target.authFile, source.authFile);
  mergeOptional(target.authUrl, source.authUrl);
  mergeOptional(target.username, source.username);
  mergeOptional(target.password, source.password);
  mergeOptional(target.projectId, source.projectId);
  mergeOptional(target.custom1, source.custom1);
  mergeOptional(target.custom2, source.custom2);
  mergeOptional(target.custom3, source.custom3);
  mergeOptional(target.custom4, source.custom4);
  mergeOptional(target.custom5, source.custom5);
}

[[nodiscard]] auto toV1Config(const SessionParameters& parameters)
    -> DeviceSessionConfig {
  DeviceSessionConfig config;
  config.baseUrl = parameters.baseUrl;
  config.token = parameters.token;
  if (parameters.authFile) {
    config.authFile = parameters.authFile->string();
  }
  config.authUrl = parameters.authUrl;
  config.username = parameters.username;
  config.password = parameters.password;
  config.custom1 = parameters.custom1;
  config.custom2 = parameters.custom2;
  config.custom3 = parameters.custom3;
  config.custom4 = parameters.custom4;
  config.custom5 = parameters.custom5;
  return config;
}
} // namespace

DeviceRegistry::DeviceRegistry(const ConfigOptions& options) {
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

  for (const auto& file : discoverFiles(options)) {
    if (file.filename() == "pyproject.toml") {
      if (auto config = readPyproject(file)) {
        mergePatches(parseConfiguration(*config, file, file.parent_path()));
      }
    } else {
      mergePatches(
          parseConfiguration(readJson(file), file, file.parent_path()));
    }
  }
  const auto inlineBase =
      options.baseDirectory.value_or(std::filesystem::current_path());
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
  if (options.inlineOverrides) {
    mergePatches(
        parseConfiguration(*options.inlineOverrides, "<inline>", inlineBase));
  }
  for (const auto& definition : options.runtimeOverrides) {
    auto patch = toPatch(definition, inlineBase);
    if (auto it = merged.find(patch.id); it != merged.end()) {
      mergePatch(it->second, patch);
    } else {
      merged.emplace(patch.id, std::move(patch));
    }
  }
  for (const auto& [unused, patch] : merged) {
    static_cast<void>(unused);
    if (auto definition = materialize(patch)) {
      definitions_.emplace_back(std::move(*definition));
    }
  }
}

void DeviceRegistry::registerDevice(DeviceDefinition definition,
                                    const bool replace) {
  if (definition.id.empty() || definition.library.empty() ||
      definition.prefix.empty()) {
    throw std::invalid_argument(
        "A device definition requires a non-empty id, library, and prefix");
  }
  const auto it =
      std::ranges::find(definitions_, definition.id, &DeviceDefinition::id);
  if (it != definitions_.end()) {
    if (!replace) {
      throw std::invalid_argument("Device '" + definition.id +
                                  "' is already registered");
    }
    *it = std::move(definition);
  } else if (definition.enabled) {
    definitions_.emplace_back(std::move(definition));
  }
  std::ranges::sort(definitions_, {}, &DeviceDefinition::id);
}

bool DeviceRegistry::unregisterDevice(const std::string_view id) {
  const auto oldSize = definitions_.size();
  std::erase_if(definitions_, [id](const DeviceDefinition& definition) {
    return definition.id == id;
  });
  return definitions_.size() != oldSize;
}

struct DeviceManager::Impl {
  explicit Impl(DeviceRegistry initialRegistry)
      : registry(std::move(initialRegistry)) {}

  DeviceRegistry registry;
  std::mutex mutex;
  std::map<std::string, std::weak_ptr<DeviceLibrary>> libraries;
};

DeviceManager::DeviceManager(const ConfigOptions& options)
    : DeviceManager(DeviceRegistry(options)) {}

DeviceManager::DeviceManager(DeviceRegistry registry)
    : impl_(std::make_unique<Impl>(std::move(registry))) {}

DeviceManager::~DeviceManager() = default;
DeviceManager::DeviceManager(DeviceManager&&) noexcept = default;
DeviceManager& DeviceManager::operator=(DeviceManager&&) noexcept = default;

const std::vector<DeviceDefinition>& DeviceManager::definitions() const {
  return impl_->registry.definitions();
}

void DeviceManager::registerDevice(DeviceDefinition definition,
                                   const bool replace) {
  impl_->registry.registerDevice(std::move(definition), replace);
}

bool DeviceManager::unregisterDevice(const std::string_view id) {
  return impl_->registry.unregisterDevice(id);
}

Device DeviceManager::open(const std::string_view id,
                           const SessionParameters& sessionOverrides) {
  const auto& definitions = impl_->registry.definitions();
  const auto definition =
      std::ranges::find(definitions, id, &DeviceDefinition::id);
  if (definition == definitions.end()) {
    throw std::out_of_range("No QDMI device is registered with id '" +
                            std::string(id) + "'");
  }
  const auto key = definition->library.string() + "\n" + definition->prefix;
  std::shared_ptr<DeviceLibrary> library;
  {
    const std::scoped_lock lock(impl_->mutex);
    library = impl_->libraries[key].lock();
    if (!library) {
      library = std::make_shared<DynamicDeviceLibrary>(
          definition->library.string(), definition->prefix);
      impl_->libraries[key] = library;
    }
  }
  auto parameters = definition->session;
  overlay(parameters, sessionOverrides);
  auto state =
      std::make_shared<QDMI_Device_impl_d>(library, toV1Config(parameters));
  auto* const handle = state.get();
  return Device(handle, std::move(state));
}

OpenAllResult
DeviceManager::openAll(const SessionParameters& sessionOverrides) {
  OpenAllResult result;
  for (const auto& definition : definitions()) {
    try {
      result.devices.emplace(definition.id,
                             open(definition.id, sessionOverrides));
    } catch (const std::exception& error) {
      result.errors.emplace(definition.id, error.what());
    }
  }
  return result;
}
} // namespace qdmi
