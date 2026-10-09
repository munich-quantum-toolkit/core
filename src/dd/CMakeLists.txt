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

  add_mqt_core_library(${MQT_CORE_TARGET_NAME}-dd ALIAS_NAME DD FORCE_STATIC)
  target_compile_definitions(${MQT_CORE_TARGET_NAME}-dd PUBLIC MQT_CORE_DD_STATIC_DEFINE)

  set(dd_targets ${MQT_CORE_TARGET_NAME}-dd)
  set(dd_export_target ${MQT_CORE_TARGET_NAME}-dd)
  if(BUILD_MQT_CORE_BINDINGS)
    # Python's DD and MLIR modules exchange objects containing addresses of DD constants.
    add_mqt_core_library(${MQT_CORE_TARGET_NAME}-dd-python ALIAS_NAME DDPython FORCE_SHARED)
    list(APPEND dd_targets ${MQT_CORE_TARGET_NAME}-dd-python)
    set(dd_export_target ${MQT_CORE_TARGET_NAME}-dd-python)
    set_target_properties(${dd_export_target} PROPERTIES DEFINE_SYMBOL mqt_core_dd_python_EXPORTS)
    install(
      TARGETS ${dd_export_target}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Python
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Python
      ARCHIVE EXCLUDE_FROM_ALL)
  endif()

  include(GenerateExportHeader)
  generate_export_header(${dd_export_target} BASE_NAME mqt_core_dd EXPORT_FILE_NAME
                         mqt_core_dd_export.h)
  foreach(dd_target IN LISTS dd_targets)
    target_sources(${dd_target} PRIVATE ${DD_SOURCES})
    target_sources(
      ${dd_target}
      PUBLIC FILE_SET
             HEADERS
             BASE_DIRS
             ${MQT_CORE_INCLUDE_BUILD_DIR}
             FILES
             ${DD_HEADERS}
             FILE_SET
             HEADERS
             BASE_DIRS
             ${CMAKE_CURRENT_BINARY_DIR}/..
             FILES
             ${CMAKE_CURRENT_BINARY_DIR}/mqt_core_dd_export.h)
    target_link_libraries(${dd_target} PRIVATE $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>)
  endforeach()

  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${MQT_CORE_TARGET_NAME}-dd
      PARENT_SCOPE)
endif()
