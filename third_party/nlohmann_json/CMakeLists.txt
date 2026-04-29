# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_CORE_NLOHMANN_JSON_VERSION
    3.12.0
    CACHE STRING "Vendored nlohmann_json version")

set(MQT_CORE_NLOHMANN_JSON_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MQT_CORE_NLOHMANN_JSON_SINGLE_INCLUDE_DIR "${MQT_CORE_NLOHMANN_JSON_ROOT_DIR}/single_include")
set(MQT_CORE_NLOHMANN_JSON_INCLUDE_INSTALL_DIR
    "${CMAKE_INSTALL_INCLUDEDIR}/mqt-core/third_party/nlohmann_json/single_include")
set(MQT_CORE_NLOHMANN_JSON_TARGET "mqt-core-vendor-nlohmann-json")

if(NOT EXISTS "${MQT_CORE_NLOHMANN_JSON_SINGLE_INCLUDE_DIR}/nlohmann/json.hpp")
  message(FATAL_ERROR "Vendored nlohmann_json header not found at "
                      "${MQT_CORE_NLOHMANN_JSON_SINGLE_INCLUDE_DIR}/nlohmann/json.hpp")
endif()

if(NOT TARGET ${MQT_CORE_NLOHMANN_JSON_TARGET})
  add_library(${MQT_CORE_NLOHMANN_JSON_TARGET} INTERFACE)
  add_library(MQT::nlohmann_json ALIAS ${MQT_CORE_NLOHMANN_JSON_TARGET})
  set_target_properties(${MQT_CORE_NLOHMANN_JSON_TARGET} PROPERTIES EXPORT_NAME nlohmann_json)

  target_include_directories(
    ${MQT_CORE_NLOHMANN_JSON_TARGET} SYSTEM
    INTERFACE $<BUILD_INTERFACE:${MQT_CORE_NLOHMANN_JSON_SINGLE_INCLUDE_DIR}>
              $<INSTALL_INTERFACE:${MQT_CORE_NLOHMANN_JSON_INCLUDE_INSTALL_DIR}>)
endif()

if(MQT_CORE_INSTALL)
  install(
    TARGETS ${MQT_CORE_NLOHMANN_JSON_TARGET}
    EXPORT ${MQT_CORE_THIRD_PARTY_TARGETS_EXPORT_NAME}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  install(
    DIRECTORY ${MQT_CORE_NLOHMANN_JSON_SINGLE_INCLUDE_DIR}/
    DESTINATION ${MQT_CORE_NLOHMANN_JSON_INCLUDE_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  if(EXISTS "${MQT_CORE_NLOHMANN_JSON_ROOT_DIR}/LICENSE.MIT")
    install(
      FILES ${MQT_CORE_NLOHMANN_JSON_ROOT_DIR}/LICENSE.MIT
      DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/mqt-core/nlohmann_json
      COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
  endif()
endif()
