/*
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <https://www.gnu.org/licenses/>.
 */

/// @file spank.cpp
/// @brief Transport administrator-declared configuration references to QDMI
/// jobs.

#include <algorithm>
#include <cstddef>
#include <exception>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" {
#include <slurm/slurm_errno.h>
#include <slurm/slurm_version.h>
#include <slurm/spank.h>
}

static_assert(SLURM_VERSION_NUMBER >= SLURM_VERSION_NUM(25, 11, 0),
              "The MQT Core SPANK component requires Slurm 25.11 or newer");

// Names and signatures in this block are fixed by Slurm's plugin ABI.
// NOLINTBEGIN(readability-identifier-naming)
extern "C" {
extern const char plugin_name[] = "mqt_core_qdmi";
extern const char plugin_type[] = "spank";
extern const unsigned int plugin_version = SLURM_VERSION_NUMBER;
extern const unsigned int spank_plugin_version = 1;
int slurm_spank_init_failure_mode =
    ESPANK_JOB_FAILURE; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
}
// NOLINTEND(readability-identifier-naming)

namespace {
constexpr auto LICENSE_ENVIRONMENT = "SLURM_JOB_LICENSES";
constexpr auto CATALOGUE_ENVIRONMENT = "MQT_CORE_QDMI_CONFIG_FILE";
constexpr size_t MAX_VALUE_SIZE = 4095;
constexpr int TASK_REJECTED = -1;

void fail(const char* message) {
  // Slurm exposes logging through a variadic C ABI.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  slurm_spank_log("mqt-core-qdmi: %s", message);
}

bool validValue(const std::string_view value) {
  return !value.empty() && value.size() <= MAX_VALUE_SIZE &&
         value.find_first_of("\r\n") == std::string_view::npos;
}

auto jobEnvironment(spank_t spank, const char* name)
    -> std::optional<std::string> {
  // S_JOB_ENV requires a mutable char*** output parameter.
  char** values = nullptr; // NOLINT(misc-const-correctness)
  // Slurm exposes item lookup through a variadic C ABI.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  if (spank_get_item(spank, S_JOB_ENV, &values) != ESPANK_SUCCESS ||
      values == nullptr) {
    throw std::runtime_error("could not read the job environment");
  }
  const auto prefix = std::string{name} + '=';
  // S_JOB_ENV is a borrowed, null-terminated array owned by Slurm.
  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  for (size_t index = 0; values[index] != nullptr; ++index) {
    const std::string_view entry{values[index]};
    if (entry.starts_with(prefix)) {
      return std::string{entry.substr(prefix.size())};
    }
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return std::nullopt;
}

auto licenseIds(std::string_view value) -> std::vector<std::string> {
  std::vector<std::string> ids;
  do {
    const auto separator = value.find(',');
    const auto id = value.substr(0, separator);
    if (!validValue(id) ||
        id.find_first_of(" \t:|@") != std::string_view::npos ||
        std::ranges::find(ids, id) != ids.end()) {
      throw std::runtime_error("invalid or duplicate configured license ID");
    }
    ids.emplace_back(id);
    if (separator == std::string_view::npos) {
      return ids;
    }
    value.remove_prefix(separator + 1);
  } while (true);
}

bool validEnvironmentName(const std::string_view name) {
  constexpr std::string_view letters =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_";
  constexpr std::string_view characters =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_0123456789";
  return !name.empty() &&
         letters.find(name.front()) != std::string_view::npos &&
         name.find_first_not_of(characters) == std::string_view::npos;
}

struct Reference {
  std::string environment;
  std::vector<std::string> licenses;
  std::string defaultValue;
};

class Configuration final {
public:
  void parse(const int count, char** arguments) {
    for (const auto* rawArgument :
         std::span{arguments, static_cast<size_t>(count)}) {
      if (rawArgument == nullptr || !validValue(rawArgument)) {
        throw std::runtime_error("malformed plugstack argument");
      }
      const std::string_view argument{rawArgument};
      const auto separator = argument.find('=');
      if (separator == std::string_view::npos) {
        throw std::runtime_error("plugstack arguments must use key=value");
      }
      const auto key = argument.substr(0, separator);
      const auto value = argument.substr(separator + 1);
      if (key == "licenses" && licenses_.empty()) {
        licenses_ = licenseIds(value);
      } else if (key == "qdmi_config_file" && !catalogueDefault_) {
        if (!validValue(value)) {
          throw std::runtime_error("invalid catalogue reference");
        }
        catalogueDefault_ = value;
      } else if (key == "reference") {
        const auto nameEnd = value.find(':');
        const auto idsEnd =
            value.find(':', nameEnd == std::string_view::npos ? value.size()
                                                              : nameEnd + 1);
        if (nameEnd == std::string_view::npos ||
            idsEnd == std::string_view::npos) {
          throw std::runtime_error("reference must use ENV:LICENSES:DEFAULT");
        }
        const auto name = value.substr(0, nameEnd);
        if (!validEnvironmentName(name) || name == CATALOGUE_ENVIRONMENT ||
            name == LICENSE_ENVIRONMENT ||
            std::ranges::any_of(references_, [&](const auto& reference) {
              return reference.environment == name;
            })) {
          throw std::runtime_error("invalid or duplicate reference name");
        }
        const auto defaultValue = value.substr(idsEnd + 1);
        if (!validValue(defaultValue)) {
          throw std::runtime_error("invalid default reference value");
        }
        references_.push_back({
            .environment = std::string{name},
            .licenses =
                licenseIds(value.substr(nameEnd + 1, idsEnd - nameEnd - 1)),
            .defaultValue = std::string{defaultValue},
        });
      } else {
        throw std::runtime_error("unknown or repeated plugstack argument");
      }
    }
    if (licenses_.empty()) {
      throw std::runtime_error("configure concrete license IDs with licenses=");
    }
    for (const auto& reference : references_) {
      for (const auto& id : reference.licenses) {
        if (std::ranges::find(licenses_, id) == licenses_.end()) {
          throw std::runtime_error("reference names an unconfigured license");
        }
      }
    }
  }

  void inject(spank_t spank) const {
    const auto expression = jobEnvironment(spank, LICENSE_ENVIRONMENT);
    const auto selected = std::ranges::find_if(licenses_, [&](const auto& id) {
      return expression && (*expression == id || *expression == id + ":1");
    });
    if (selected == licenses_.end()) {
      return;
    }

    if (catalogueDefault_) {
      apply(spank, CATALOGUE_ENVIRONMENT, *catalogueDefault_);
    }
    for (const auto& reference : references_) {
      if (std::ranges::find(reference.licenses, *selected) !=
          reference.licenses.end()) {
        apply(spank, reference.environment.c_str(), reference.defaultValue);
      }
    }
  }

private:
  static void apply(spank_t spank, const char* environment,
                    const std::string& defaultValue) {
    const auto value = jobEnvironment(spank, environment);
    if (value && !validValue(*value)) {
      throw std::runtime_error(
          "job environment value is malformed or too long");
    }
    if (value) {
      return;
    }
    const auto result =
        spank_setenv(spank, environment, defaultValue.c_str(), 0);
    if (result != ESPANK_SUCCESS && result != ESPANK_ENV_EXISTS) {
      throw std::runtime_error("failed to set a default QDMI reference");
    }
  }

  std::vector<std::string> licenses_;
  std::vector<Reference> references_;
  std::optional<std::string> catalogueDefault_;
};

Configuration
    configuration; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool licenseEnvironmentReady = true;

} // namespace

// Names and signatures are fixed by Slurm's plugin ABI.
// NOLINTBEGIN(misc-use-internal-linkage, readability-identifier-naming,
// cppcoreguidelines-avoid-c-arrays)
extern "C" {
int slurm_spank_init(spank_t /*spank*/, const int count, char* arguments[]) {
  try {
    configuration = Configuration{};
    licenseEnvironmentReady = true;
    configuration.parse(count, arguments);
    return ESPANK_SUCCESS;
  } catch (const std::exception& error) {
    fail(error.what());
  } catch (...) {
    fail("QDMI SPANK initialization failed");
  }
  return TASK_REJECTED;
}

int slurm_spank_user_init(spank_t spank, int /*count*/, char* /*arguments*/[]) {
  if (spank_remote(spank) == 1) {
    // Slurm restores allocated licenses before task-init. A submitted value
    // must not make a job with no allocated license match a configured device.
    const auto result = spank_unsetenv(spank, LICENSE_ENVIRONMENT);
    if (result != ESPANK_SUCCESS && result != ESPANK_ENV_NOEXIST) {
      // Fail in task-init: a user-init error can drain the node.
      licenseEnvironmentReady = false;
      fail("could not clear inherited license selection");
    }
  }
  return ESPANK_SUCCESS;
}

int slurm_spank_task_init(spank_t spank, int /*count*/, char* /*arguments*/[]) {
  if (spank_remote(spank) != 1) {
    return ESPANK_SUCCESS;
  }
  try {
    if (!licenseEnvironmentReady) {
      throw std::runtime_error("could not clear inherited license selection");
    }
    configuration.inject(spank);
    return ESPANK_SUCCESS;
  } catch (const std::exception& error) {
    fail(error.what());
  } catch (...) {
    fail("QDMI configuration injection failed");
  }
  return TASK_REJECTED;
}
}
// NOLINTEND(misc-use-internal-linkage, readability-identifier-naming,
// cppcoreguidelines-avoid-c-arrays)
