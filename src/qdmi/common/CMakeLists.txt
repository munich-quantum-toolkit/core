# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-common)

if(NOT TARGET ${TARGET_NAME})
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMICommon)

  target_sources(${TARGET_NAME} PRIVATE Common.cpp DeviceConfiguration.cpp)

  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/common/Common.hpp
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/common/DeviceConfiguration.hpp
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/common/Diagnostics.hpp)

  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi
    PRIVATE qdmi::qdmi_project_warnings ${CMAKE_DL_LIBS})

  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
