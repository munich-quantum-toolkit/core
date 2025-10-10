cmake_minimum_required(VERSION 3.5...4.0)

#
# PROJECT name and version
#
project(
  nlohmann_json
  VERSION 3.12.0
  LANGUAGES CXX)

#
# INCLUDE
#
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(ExternalProject)

#
# OPTIONS
#

if(POLICY CMP0077)
  # Allow CMake 3.13+ to override options when using FetchContent / add_subdirectory.
  cmake_policy(SET CMP0077 NEW)
endif()

option(JSON_Install "Install CMake targets during install step." ${MQT_CORE_INSTALL})
option(JSON_SystemInclude "Include as system headers (skip for clang-tidy)." ON)

#
# CONFIGURATION
#
include(GNUInstallDirs)

if(NOT DEFINED NLOHMANN_JSON_TARGET_NAME)
  # Allow overriding the target name when using FetchContent / add_subdirectory.
  set(NLOHMANN_JSON_TARGET_NAME ${PROJECT_NAME})
endif()

set(NLOHMANN_JSON_CONFIG_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}"
    CACHE INTERNAL "")
set(NLOHMANN_JSON_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
set(NLOHMANN_JSON_TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(NLOHMANN_JSON_CMAKE_CONFIG_TEMPLATE "cmake/config.cmake.in")
set(NLOHMANN_JSON_CMAKE_CONFIG_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(NLOHMANN_JSON_CMAKE_VERSION_CONFIG_FILE
    "${NLOHMANN_JSON_CMAKE_CONFIG_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(NLOHMANN_JSON_CMAKE_PROJECT_CONFIG_FILE
    "${NLOHMANN_JSON_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Config.cmake")
set(NLOHMANN_JSON_CMAKE_PROJECT_TARGETS_FILE
    "${NLOHMANN_JSON_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Targets.cmake")
set(NLOHMANN_JSON_PKGCONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/pkgconfig")

set(NLOHMANN_JSON_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/single_include/")

if(JSON_SystemInclude)
  set(NLOHMANN_JSON_SYSTEM_INCLUDE "SYSTEM")
endif()

#
# TARGET create target and add include path
#
add_library(${NLOHMANN_JSON_TARGET_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${NLOHMANN_JSON_TARGET_NAME} ALIAS ${NLOHMANN_JSON_TARGET_NAME})
target_compile_features(${NLOHMANN_JSON_TARGET_NAME} INTERFACE cxx_std_11)

target_include_directories(
  ${NLOHMANN_JSON_TARGET_NAME} ${NLOHMANN_JSON_SYSTEM_INCLUDE}
  INTERFACE $<BUILD_INTERFACE:${NLOHMANN_JSON_INCLUDE_BUILD_DIR}>
            $<INSTALL_INTERFACE:${NLOHMANN_JSON_INCLUDE_INSTALL_DIR}>)

# Install a pkg-config file, so other tools can find this.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.pc.in"
               "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" @ONLY)

#
# INSTALL install header files, generate and install cmake config files for find_package()
#
include(CMakePackageConfigHelpers)
# use a custom package version config file instead of write_basic_package_version_file to ensure
# that it's architecture-independent https://github.com/nlohmann/json/issues/1697
configure_file("cmake/nlohmann_jsonConfigVersion.cmake.in"
               ${NLOHMANN_JSON_CMAKE_VERSION_CONFIG_FILE} @ONLY)
configure_file(${NLOHMANN_JSON_CMAKE_CONFIG_TEMPLATE} ${NLOHMANN_JSON_CMAKE_PROJECT_CONFIG_FILE}
               @ONLY)

if(JSON_Install)
  install(DIRECTORY ${NLOHMANN_JSON_INCLUDE_BUILD_DIR}
          DESTINATION ${NLOHMANN_JSON_INCLUDE_INSTALL_DIR})
  install(FILES ${NLOHMANN_JSON_CMAKE_PROJECT_CONFIG_FILE}
                ${NLOHMANN_JSON_CMAKE_VERSION_CONFIG_FILE}
          DESTINATION ${NLOHMANN_JSON_CONFIG_INSTALL_DIR})
  export(
    TARGETS ${NLOHMANN_JSON_TARGET_NAME}
    NAMESPACE ${PROJECT_NAME}::
    FILE ${NLOHMANN_JSON_CMAKE_PROJECT_TARGETS_FILE})
  install(
    TARGETS ${NLOHMANN_JSON_TARGET_NAME}
    EXPORT ${NLOHMANN_JSON_TARGETS_EXPORT_NAME}
    INCLUDES
    DESTINATION ${NLOHMANN_JSON_INCLUDE_INSTALL_DIR})
  install(
    EXPORT ${NLOHMANN_JSON_TARGETS_EXPORT_NAME}
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${NLOHMANN_JSON_CONFIG_INSTALL_DIR})
  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
          DESTINATION ${NLOHMANN_JSON_PKGCONFIG_INSTALL_DIR})
  include(CPack)
endif()
