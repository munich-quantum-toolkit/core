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
  std::string optionName;
  std::vector<std::string> licenses;
  std::optional<std::string> defaultValue;
  std::optional<std::string> option;
};

int optionCallback(int value, const char* argument, int remote);

// The Slurm ABI uses mutable pointers for option metadata.
char* optionText(const char* value) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<char*>(value);
}

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
        if (!validEnvironmentName(name) ||
            name.size() + std::string_view{"qdmi-ref-"}.size() >=
                SPANK_OPTION_MAXLEN ||
            name == CATALOGUE_ENVIRONMENT || name == LICENSE_ENVIRONMENT ||
            std::ranges::any_of(references_, [&](const auto& reference) {
              return reference.environment == name;
            })) {
          throw std::runtime_error("invalid or duplicate reference name");
        }
        const auto defaultValue = value.substr(idsEnd + 1);
        if (!defaultValue.empty() && !validValue(defaultValue)) {
          throw std::runtime_error("invalid default reference value");
        }
        references_.push_back({
            .environment = std::string{name},
            .optionName = "qdmi-ref-" + std::string{name},
            .licenses =
                licenseIds(value.substr(nameEnd + 1, idsEnd - nameEnd - 1)),
            .defaultValue = defaultValue.empty()
                                ? std::nullopt
                                : std::make_optional(std::string{defaultValue}),
            .option = std::nullopt,
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

  void registerOptions(spank_t spank) const {
    spank_option option{
        .name = optionText("qdmi-config-file"),
        .arginfo = optionText("PATH"),
        .usage = optionText("QDMI device catalogue path"),
        .has_arg = 1,
        .val = 0,
        .cb = optionCallback,
    };
    if (spank_option_register(spank, &option) != ESPANK_SUCCESS) {
      throw std::runtime_error("failed to register the QDMI catalogue option");
    }
    for (size_t index = 0; index < references_.size(); ++index) {
      option.name = optionText(references_[index].optionName.c_str());
      option.arginfo = optionText("VALUE");
      option.usage =
          optionText("Administrator-declared configuration reference");
      option.val = static_cast<int>(index + 1);
      if (spank_option_register(spank, &option) != ESPANK_SUCCESS) {
        throw std::runtime_error("failed to register a QDMI reference option");
      }
    }
  }

  void setOption(const int option, const char* argument) {
    if (argument == nullptr || !validValue(argument)) {
      throw std::runtime_error("invalid QDMI option value");
    }
    if (option == 0) {
      catalogueOption_ = argument;
      return;
    }
    if (option < 1 || static_cast<size_t>(option) > references_.size()) {
      throw std::runtime_error("unknown QDMI reference option");
    }
    references_[static_cast<size_t>(option) - 1].option = argument;
  }

  void inject(spank_t spank) const {
    const auto expression = jobEnvironment(spank, LICENSE_ENVIRONMENT);
    std::vector<std::string_view> selected;
    if (expression) {
      std::string_view remaining{*expression};
      do {
        const auto end = remaining.find_first_of(",|");
        auto token = remaining.substr(0, end);
        const auto begin = token.find_first_not_of(" \t");
        token = begin == std::string_view::npos ? std::string_view{}
                                                : token.substr(begin);
        const auto id = token.substr(0, token.find_first_of(":@ \t"));
        if (std::ranges::find(licenses_, id) != licenses_.end()) {
          selected.push_back(id);
        }
        if (end == std::string_view::npos) {
          break;
        }
        remaining.remove_prefix(end + 1);
      } while (true);
    }
    if (selected.empty()) {
      if (catalogueOption_ ||
          std::ranges::any_of(references_, [](const auto& reference) {
            return reference.option.has_value();
          })) {
        throw std::runtime_error(
            "QDMI options require a configured device license");
      }
      return;
    }

    apply(spank, CATALOGUE_ENVIRONMENT, catalogueDefault_, catalogueOption_);
    for (const auto& reference : references_) {
      const bool applicable = std::ranges::any_of(selected, [&](const auto id) {
        return std::ranges::find(reference.licenses, id) !=
               reference.licenses.end();
      });
      if (!applicable) {
        if (reference.option) {
          throw std::runtime_error(
              "QDMI reference does not apply to this license");
        }
        continue;
      }
      apply(spank, reference.environment.c_str(), reference.defaultValue,
            reference.option);
    }
  }

private:
  static void apply(spank_t spank, const char* environment,
                    const std::optional<std::string>& defaultValue,
                    const std::optional<std::string>& option) {
    if (option) {
      if (spank_setenv(spank, environment, option->c_str(), 1) !=
          ESPANK_SUCCESS) {
        throw std::runtime_error("failed to set an explicit QDMI reference");
      }
      return;
    }
    const auto value = jobEnvironment(spank, environment);
    if (value && !validValue(*value)) {
      throw std::runtime_error(
          "job environment value is malformed or too long");
    }
    if (value || !defaultValue) {
      return;
    }
    const auto result =
        spank_setenv(spank, environment, defaultValue->c_str(), 0);
    if (result != ESPANK_SUCCESS && result != ESPANK_ENV_EXISTS) {
      throw std::runtime_error("failed to set a default QDMI reference");
    }
  }

  std::vector<std::string> licenses_;
  std::vector<Reference> references_;
  std::optional<std::string> catalogueDefault_;
  std::optional<std::string> catalogueOption_;
};

Configuration
    configuration; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
bool licenseEnvironmentReady = true;

int optionCallback(const int value, const char* argument, int /*remote*/) {
  try {
    configuration.setOption(value, argument);
    return ESPANK_SUCCESS;
  } catch (const std::exception& error) {
    fail(error.what());
  } catch (...) {
    fail("QDMI option processing failed");
  }
  return TASK_REJECTED;
}

} // namespace

// Names and signatures are fixed by Slurm's plugin ABI.
// NOLINTBEGIN(misc-use-internal-linkage, readability-identifier-naming,
// cppcoreguidelines-avoid-c-arrays)
extern "C" {
int slurm_spank_init(spank_t spank, const int count, char* arguments[]) {
  try {
    configuration = Configuration{};
    licenseEnvironmentReady = true;
    configuration.parse(count, arguments);
    configuration.registerOptions(spank);
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
