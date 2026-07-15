# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(common)
add_subdirectory(devices)

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi)

if(NOT TARGET ${TARGET_NAME})
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMI)

  target_sources(${TARGET_NAME} PRIVATE Device.cpp DeviceApi.cpp DeviceApi.hpp DeviceManager.cpp
                                        DeviceRegistry.cpp V1DeviceApi.cpp)

  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/Device.hpp
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/DeviceManager.hpp
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/DeviceRegistry.hpp)

  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC nlohmann_json::nlohmann_json
    PRIVATE qdmi::qdmi MQT::CoreQDMICommon spdlog::spdlog ${CMAKE_DL_LIBS})
  target_include_directories(${TARGET_NAME} SYSTEM
                             PRIVATE ${PROJECT_SOURCE_DIR}/vendor/tomlplusplus)

  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
