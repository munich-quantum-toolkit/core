# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_CORE_TARGETS "")

if(MQT_CORE_INSTALL)
  include(GNUInstallDirs)
  set(MQT_CORE_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/mqt-core")
  set(MQT_CORE_TARGETS_EXPORT_NAME "mqt-core-targets")
  set(MQT_CORE_CMAKE_CONFIG_TEMPLATE "${PROJECT_SOURCE_DIR}/cmake/mqt-core-config.cmake.in")
  set(MQT_CORE_CMAKE_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  set(MQT_CORE_CMAKE_VERSION_CONFIG_FILE
      "${MQT_CORE_CMAKE_CONFIG_DIR}/mqt-core-config-version.cmake")
  set(MQT_CORE_CMAKE_PROJECT_CONFIG_FILE "${MQT_CORE_CMAKE_CONFIG_DIR}/mqt-core-config.cmake")
  set(MQT_CORE_CMAKE_PROJECT_TARGETS_FILE "${MQT_CORE_CMAKE_CONFIG_DIR}/mqt-core-targets.cmake")
endif()

if(NOT TARGET MQT::ProjectWarnings)
  # use the warnings specified in CompilerWarnings.cmake
  add_library(project_warnings INTERFACE)

  # standard compiler warnings
  include(${PROJECT_SOURCE_DIR}/cmake/CompilerWarnings.cmake)
  set_project_warnings(project_warnings)

  # add MQT alias
  add_library(MQT::ProjectWarnings ALIAS project_warnings)
  set_target_properties(project_warnings PROPERTIES EXPORT_NAME ProjectWarnings)

  # add to list of MQT core targets
  list(APPEND MQT_CORE_TARGETS project_warnings)
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

  # add MQT alias
  add_library(MQT::ProjectOptions ALIAS project_options)
  set_target_properties(project_options PROPERTIES EXPORT_NAME ProjectOptions)

  # add to list of MQT core targets
  list(APPEND MQT_CORE_TARGETS project_options)
endif()

# add the IR package
add_subdirectory(ir)

# add the QASM package
add_subdirectory(qasm3)

# add the algorithms package
add_subdirectory(algorithms)

# add the circuit_optimizer package
add_subdirectory(circuit_optimizer)

# add datastructures package
add_subdirectory(datastructures)

# add DD package library
add_subdirectory(dd)

# add ZX package library
add_subdirectory(zx)

# add NA library
add_subdirectory(na)

# add QDMI package library
add_subdirectory(qdmi)

# add FoMaC package library
add_subdirectory(fomac)

# add QIR package library
add_subdirectory(qir)

# Installation instructions for the main library
if(MQT_CORE_INSTALL)
  include(CMakePackageConfigHelpers)
  configure_package_config_file(
    ${MQT_CORE_CMAKE_CONFIG_TEMPLATE} ${MQT_CORE_CMAKE_PROJECT_CONFIG_FILE}
    INSTALL_DESTINATION ${MQT_CORE_CONFIG_INSTALL_DIR}
    PATH_VARS MQT_CORE_WITH_GMP
    NO_SET_AND_CHECK_MACRO NO_CHECK_REQUIRED_COMPONENTS_MACRO)
  write_basic_package_version_file(
    ${MQT_CORE_CMAKE_VERSION_CONFIG_FILE}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion)

  install(
    FILES ${MQT_CORE_CMAKE_PROJECT_CONFIG_FILE} ${MQT_CORE_CMAKE_VERSION_CONFIG_FILE}
    DESTINATION ${MQT_CORE_CONFIG_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  # FILE_SET should be aligned with LIBRARY and ARCHIVE, so we turn cmake-format off
  # cmake-format: off
  install(
    TARGETS ${MQT_CORE_TARGETS}
    EXPORT ${MQT_CORE_TARGETS_EXPORT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
            NAMELINK_COMPONENT ${MQT_CORE_TARGET_NAME}_Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${MQT_CORE_TARGET_NAME}_Development
    FILE_SET HEADERS
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mqt-core
            COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
  # cmake-format: on

  install(
    EXPORT ${MQT_CORE_TARGETS_EXPORT_NAME}
    FILE ${MQT_CORE_PROJECT_TARGETS_FILE}
    NAMESPACE MQT::
    DESTINATION ${MQT_CORE_CONFIG_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  install(
    FILES ${PROJECT_SOURCE_DIR}/cmake/AddMQTPythonBinding.cmake
          ${PROJECT_SOURCE_DIR}/cmake/Cache.cmake
          ${PROJECT_SOURCE_DIR}/cmake/FindGMP.cmake
          ${PROJECT_SOURCE_DIR}/cmake/PackageAddTest.cmake
          ${PROJECT_SOURCE_DIR}/cmake/PreventInSourceBuilds.cmake
          ${PROJECT_SOURCE_DIR}/cmake/StandardProjectSettings.cmake
    DESTINATION ${MQT_CORE_CONFIG_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  # Install nlohmann_json as part of the Development component
  if(JSON_Install)
    install(
      TARGETS nlohmann_json
      EXPORT ${MQT_CORE_TARGETS_EXPORT_NAME}
      INCLUDES
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
  endif()
endif()
