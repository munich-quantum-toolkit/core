# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET MQT::nlohmann_json)
  FetchContent_GetProperties(nlohmann_json)
  if(nlohmann_json_POPULATED)
    set(NLOHMANN_JSON_INCLUDE_DIR "${nlohmann_json_SOURCE_DIR}/single_include")
  else()
    message(FATAL_ERROR "nlohmann_json not found!")
  endif()
  message(STATUS "Using nlohmann_json from ${NLOHMANN_JSON_INCLUDE_DIR}")

  add_library(mqt-core-vendored-nlohmann_json INTERFACE)
  target_include_directories(mqt-core-vendored-nlohmann_json SYSTEM
                             INTERFACE $<BUILD_INTERFACE:${NLOHMANN_JSON_INCLUDE_DIR}>)
  add_library(MQT::nlohmann_json ALIAS mqt-core-vendored-nlohmann_json)

  # add headers to file sets
  file(GLOB_RECURSE NLOHMANN_JSON_HEADERS ${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/*.hpp)
  target_sources(
    mqt-core-vendored-nlohmann_json PUBLIC FILE_SET HEADERS BASE_DIRS ${NLOHMANN_JSON_INCLUDE_DIR}
                                           FILES ${NLOHMANN_JSON_HEADERS})

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} mqt-core-vendored-nlohmann_json
      PARENT_SCOPE)
endif()
