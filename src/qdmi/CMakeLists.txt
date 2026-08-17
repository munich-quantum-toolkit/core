# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(common)
add_subdirectory(devices)
add_subdirectory(driver)

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi)

if(NOT TARGET ${TARGET_NAME})
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMI)

  target_sources(${TARGET_NAME} PRIVATE Client.cpp Slurm.cpp)

  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/Client.hpp
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/Slurm.hpp)

  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreQDMICommon MQT::CoreQDMIDriver
    PRIVATE spdlog::spdlog)

  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
