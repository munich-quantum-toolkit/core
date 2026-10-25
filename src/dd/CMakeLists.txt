# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT TARGET ${MQT_CORE_TARGET_NAME}-dd)
  file(GLOB_RECURSE DD_HEADERS ${MQT_CORE_INCLUDE_BUILD_DIR}/dd/*.hpp)
  file(GLOB_RECURSE DD_SOURCES **.cpp)

  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-dd ALIAS_NAME DD FORCE_STATIC HIDDEN_VISIBILITY)
  target_sources(
    ${MQT_CORE_TARGET_NAME}-dd
    PRIVATE ${DD_SOURCES}
    PUBLIC
    FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR} FILES ${DD_HEADERS})
  target_link_libraries(${MQT_CORE_TARGET_NAME}-dd PUBLIC LLVMSupport)
  set(MQT_CORE_TARGETS ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-dd PARENT_SCOPE)
endif()
