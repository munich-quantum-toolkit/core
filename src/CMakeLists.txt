# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_CORE_TARGETS "")

if(NOT TARGET MQT::ProjectWarnings)
  # use the warnings specified in CompilerWarnings.cmake
  add_library(project_warnings INTERFACE)

  # standard compiler warnings
  include(${PROJECT_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
  set_project_warnings(project_warnings)

  add_library(MQT::ProjectWarnings ALIAS project_warnings)
  set_target_properties(project_warnings PROPERTIES EXPORT_NAME ProjectWarnings)
endif()

if(NOT TARGET MQT::ProjectOptions)
  # use the options specified in CompilerOptions.cmake
  add_library(project_options INTERFACE)

  # standard compiler options
  include(${PROJECT_SOURCE_DIR}/cmake/CompilerOptions.cmake)
  enable_project_options(project_options)

  # sanitizer options if supported by compiler
  include(${PROJECT_SOURCE_DIR}/cmake/Sanitizers.cmake)
  enable_sanitizers(project_options)

  add_library(MQT::ProjectOptions ALIAS project_options)
  set_target_properties(project_options PROPERTIES EXPORT_NAME ProjectOptions)
endif()

add_subdirectory(bench)

add_subdirectory(dd)

add_subdirectory(qdmi)

if(MQT_CORE_INSTALL)
  include(CMakePackageConfigHelpers)
  set(config_install_dir "${CMAKE_INSTALL_DATADIR}/cmake/mqt-core")
  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/mqt-core-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mqt-core-config.cmake
    INSTALL_DESTINATION ${config_install_dir}
    NO_SET_AND_CHECK_MACRO)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/mqt-core-config-version.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

  mqt_get_qdmi_device_targets(runtime_targets)
  list(APPEND runtime_targets ${MQT_CORE_TARGET_NAME}-qdmi-driver)
  list(REMOVE_ITEM MQT_CORE_TARGETS ${runtime_targets})
  set_target_properties(${runtime_targets} PROPERTIES INTERFACE_COMPILE_FEATURES "")

  # Runtime exports contain only the C ABI and executable tools.
  # cmake-format: off
  install(
    TARGETS ${runtime_targets}
    EXPORT mqt-core-runtime-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mqt-core
                     COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)

  install(
    TARGETS ${MQT_CORE_TARGETS}
    EXPORT mqt-core-development-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Development
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mqt-core
                     COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
  # cmake-format: on

  install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/mqt-core-config.cmake
          ${CMAKE_CURRENT_BINARY_DIR}/mqt-core-config-version.cmake
          ${PROJECT_SOURCE_DIR}/cmake/AddMQTQDMIDevice.cmake
    DESTINATION ${config_install_dir}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
  install(
    EXPORT mqt-core-runtime-targets
    NAMESPACE MQT::
    DESTINATION ${config_install_dir}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
  install(
    EXPORT mqt-core-development-targets
    NAMESPACE MQT::
    DESTINATION ${config_install_dir}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
endif()
