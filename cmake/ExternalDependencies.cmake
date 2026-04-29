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

if(BUILD_MQT_CORE_MLIR)
  set(Eigen_VERSION
      5.0.1
      CACHE STRING "Eigen version")
  set(Eigen_URL
      https://gitlab.com/libeigen/eigen/-/archive/${Eigen_VERSION}/eigen-${Eigen_VERSION}.tar.gz)
  set(EIGEN_BUILD_TESTING
      OFF
      CACHE INTERNAL "Disable building Eigen tests")
  FetchContent_Declare(Eigen URL ${Eigen_URL} FIND_PACKAGE_ARGS ${Eigen_VERSION})
  list(APPEND FETCH_PACKAGES Eigen)

  # Fetch jeff-mlir
  set(BUILD_JEFF_MLIR_TRANSLATION
      OFF
      CACHE BOOL "Disable building the translation submodule of jeff-mlir")
  FetchContent_Declare(
    jeff-mlir
    GIT_REPOSITORY https://github.com/PennyLaneAI/jeff-mlir.git
    GIT_TAG v0.1.0)
  list(APPEND FETCH_PACKAGES jeff-mlir)
endif()

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
set(QDMI_VERSION 1.3.0
        CACHE STRING "QDMI version")
set(QDMI_REV "0f7e08c58b72800d1022a01cfb618af67b9a9c30" # v1.3.0
        CACHE STRING "QDMI identifier (tag, branch or commit hash)")
set(QDMI_REPO_OWNER "Munich-Quantum-Software-Stack"
        CACHE STRING "QDMI repository owner (change when using a fork)")
cmake_dependent_option(INSTALL_QDMI "Install QDMI library" ON "MQT_CORE_INSTALL" OFF)
# cmake-format: on
FetchContent_Declare(
  qdmi
  GIT_REPOSITORY https://github.com/${QDMI_REPO_OWNER}/qdmi.git
  GIT_TAG ${QDMI_REV}
  FIND_PACKAGE_ARGS ${QDMI_VERSION})
list(APPEND FETCH_PACKAGES qdmi)

if(FETCH_PACKAGES)
  # Make all declared dependencies available.
  FetchContent_MakeAvailable(${FETCH_PACKAGES})
endif()

# Treat Eigen headers as system headers to avoid surfacing third-party warnings.
set(_eigen_target "")
if(TARGET Eigen3::Eigen)
  set(_eigen_target Eigen3::Eigen)
elseif(TARGET Eigen::Eigen)
  set(_eigen_target Eigen::Eigen)
endif()
if(_eigen_target)
  get_target_property(_eigen_alias_target ${_eigen_target} ALIASED_TARGET)
  if(_eigen_alias_target)
    set(_eigen_target ${_eigen_alias_target})
  endif()
  get_target_property(_eigen_includes ${_eigen_target} INTERFACE_INCLUDE_DIRECTORIES)
  if(_eigen_includes)
    set_target_properties(${_eigen_target} PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                                                      "${_eigen_includes}")
  endif()
endif()

set(MQT_CORE_THIRD_PARTY_CONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/cmake/mqt-core")
set(MQT_CORE_THIRD_PARTY_TARGETS_EXPORT_NAME "mqt-core-third-party-targets")
set(MQT_CORE_THIRD_PARTY_TARGETS_FILE "mqt-core-third-party-targets.cmake")

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party ${CMAKE_CURRENT_BINARY_DIR}/third_party)

if(MQT_CORE_INSTALL)
  install(
    EXPORT ${MQT_CORE_THIRD_PARTY_TARGETS_EXPORT_NAME}
    FILE ${MQT_CORE_THIRD_PARTY_TARGETS_FILE}
    NAMESPACE MQT::
    DESTINATION ${MQT_CORE_THIRD_PARTY_CONFIG_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
endif()
