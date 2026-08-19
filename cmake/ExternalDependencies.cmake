# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Declare all external dependencies and make sure that they are available.

include(FetchContent)
include(CMakeDependentOption)
include(GNUInstallDirs)
set(FETCH_PACKAGES "")

if(BUILD_MQT_CORE_BINDINGS)
  # Detect the installed nanobind package and import it into CMake
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
endif()

# Fetch jeff-mlir
FetchContent_Declare(
  jeff-mlir
  GIT_REPOSITORY https://github.com/unitaryfoundation/jeff-mlir.git
  GIT_TAG 66c92d058cb498f5c12628f6a2d2a290480d700b)
function(_mqt_core_make_jeff_available)
  # jeff's transitive Cap'n Proto dependency contains source files that cannot share a unity
  # translation unit. Keep the complete dependency subtree out of unity builds.
  set(CMAKE_UNITY_BUILD OFF)
  FetchContent_MakeAvailable(jeff-mlir)
endfunction()
_mqt_core_make_jeff_available()

set(JSON_VERSION
    3.12.0
    CACHE STRING "nlohmann_json version")
set(JSON_URL https://github.com/nlohmann/json/releases/download/v${JSON_VERSION}/json.tar.xz)
set(JSON_SystemInclude
    ON
    CACHE INTERNAL "Treat the library headers like system headers")
FetchContent_Declare(nlohmann_json URL ${JSON_URL} FIND_PACKAGE_ARGS ${JSON_VERSION})
list(APPEND FETCH_PACKAGES nlohmann_json)

if(BUILD_MQT_CORE_TESTS)
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  # Disable the install instructions for GTest, as we do not need them.
  set(INSTALL_GTEST
      OFF
      CACHE BOOL "" FORCE)
  set(GTEST_VERSION
      1.17.0
      CACHE STRING "Google Test version")
  set(GTEST_URL https://github.com/google/googletest/archive/refs/tags/v${GTEST_VERSION}.tar.gz)
  FetchContent_Declare(googletest URL ${GTEST_URL} FIND_PACKAGE_ARGS ${GTEST_VERSION} NAMES GTest)
  list(APPEND FETCH_PACKAGES googletest)
endif()

# cmake-format: off
set(QDMI_MINIMUM_VERSION 1.3.3
        CACHE STRING "Minimum QDMI version")
set(QDMI_VERSION 1.3.3
        CACHE STRING "QDMI version")
set(QDMI_REV "18cfb67fd9042761d3005c2f8655751c1758f9c5" # v1.3.3
        CACHE STRING "QDMI identifier (tag, branch or commit hash)")
set(QDMI_REPO_OWNER "Munich-Quantum-Software-Stack"
        CACHE STRING "QDMI repository owner (change when using a fork)")
cmake_dependent_option(INSTALL_QDMI "Install QDMI library" ON "MQT_CORE_INSTALL" OFF)
# cmake-format: on
FetchContent_Declare(
  qdmi
  GIT_REPOSITORY https://github.com/${QDMI_REPO_OWNER}/qdmi.git
  GIT_TAG ${QDMI_REV}
  FIND_PACKAGE_ARGS ${QDMI_MINIMUM_VERSION})
list(APPEND FETCH_PACKAGES qdmi)

set(MQT_CORE_MANAGES_SPDLOG OFF)
if(NOT TARGET spdlog::spdlog)
  set(SPDLOG_VERSION
      1.17.0
      CACHE STRING "spdlog version")
  set(SPDLOG_URL https://github.com/gabime/spdlog/archive/refs/tags/v${SPDLOG_VERSION}.tar.gz)
  # Add position independent code for spdlog, this is required for Python bindings on Linux.
  set(SPDLOG_BUILD_PIC ON)
  set(SPDLOG_SYSTEM_INCLUDES
      ON
      CACHE INTERNAL "Treat the library headers like system headers")
  cmake_dependent_option(MQT_CORE_SPDLOG_INSTALL "Install spdlog library" ON "MQT_CORE_INSTALL" OFF)
  # Disable upstream spdlog install rules and install with explicit MQT components below.
  set(SPDLOG_INSTALL
      OFF
      CACHE BOOL "Disable upstream spdlog install rules; handled by mqt-core" FORCE)
  cmake_dependent_option(SPDLOG_BUILD_SHARED "Build spdlog as shared library" ON
                         "BUILD_MQT_CORE_SHARED_LIBS" OFF)
  FetchContent_Declare(spdlog URL ${SPDLOG_URL} FIND_PACKAGE_ARGS ${SPDLOG_VERSION})
  list(APPEND FETCH_PACKAGES spdlog)
  set(MQT_CORE_MANAGES_SPDLOG ON)
endif()

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})

# Ensure external shared libraries end up in a common lib layout used by our RUNPATH
if(MQT_CORE_MANAGES_SPDLOG AND TARGET spdlog)
  set_target_properties(
    spdlog
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
               ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
endif()

# Install spdlog with explicit MQT components.
if(MQT_CORE_MANAGES_SPDLOG
   AND MQT_CORE_SPDLOG_INSTALL
   AND TARGET spdlog
   AND TARGET spdlog_header_only)
  include(CMakePackageConfigHelpers)

  set(MQT_CORE_SPDLOG_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/spdlog")
  set(MQT_CORE_SPDLOG_CONFIG_TARGETS_FILE "spdlogConfigTargets.cmake")
  set(MQT_CORE_SPDLOG_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/spdlogConfig.cmake")
  set(MQT_CORE_SPDLOG_VERSION_CONFIG_FILE "${CMAKE_CURRENT_BINARY_DIR}/spdlogConfigVersion.cmake")

  install(
    TARGETS spdlog spdlog_header_only
    EXPORT spdlog
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime
            NAMELINK_COMPONENT ${MQT_CORE_TARGET_NAME}_Development
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  install(
    DIRECTORY ${spdlog_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development
    PATTERN "fmt/bundled" EXCLUDE)

  if(NOT SPDLOG_USE_STD_FORMAT
     AND NOT SPDLOG_FMT_EXTERNAL
     AND NOT SPDLOG_FMT_EXTERNAL_HO)
    install(
      DIRECTORY ${spdlog_SOURCE_DIR}/include/spdlog/fmt/bundled/
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/spdlog/fmt/bundled
      COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
  endif()

  install(
    EXPORT spdlog
    FILE ${MQT_CORE_SPDLOG_CONFIG_TARGETS_FILE}
    NAMESPACE spdlog::
    DESTINATION ${MQT_CORE_SPDLOG_CONFIG_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  set(config_targets_file ${MQT_CORE_SPDLOG_CONFIG_TARGETS_FILE})
  configure_package_config_file(
    ${spdlog_SOURCE_DIR}/cmake/spdlogConfig.cmake.in ${MQT_CORE_SPDLOG_CONFIG_FILE}
    INSTALL_DESTINATION ${MQT_CORE_SPDLOG_CONFIG_INSTALL_DIR})
  write_basic_package_version_file(
    ${MQT_CORE_SPDLOG_VERSION_CONFIG_FILE}
    VERSION ${SPDLOG_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(
    FILES ${MQT_CORE_SPDLOG_CONFIG_FILE} ${MQT_CORE_SPDLOG_VERSION_CONFIG_FILE}
    DESTINATION ${MQT_CORE_SPDLOG_CONFIG_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
endif()
