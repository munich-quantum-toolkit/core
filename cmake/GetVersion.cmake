# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Read the static package version for standalone and FetchContent builds.
set(version_file "${CMAKE_CURRENT_LIST_DIR}/../pyproject.toml")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${version_file}")
file(STRINGS "${version_file}" version_line LIMIT_COUNT 1 REGEX "^version = \"[0-9]")
if(NOT version_line MATCHES "^version = \"(([0-9]+[.][0-9]+[.][0-9]+)[^\"]*)\"$")
  message(FATAL_ERROR "Cannot read the MQT Core version from ${version_file}")
endif()
set(MQT_CORE_VERSION_STRING "${CMAKE_MATCH_1}")
set(MQT_CORE_VERSION "${CMAKE_MATCH_2}")
