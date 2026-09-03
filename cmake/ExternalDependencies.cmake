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
  # Fetch jeff-mlir
  FetchContent_Declare(
    jeff-mlir
    GIT_REPOSITORY https://github.com/unitaryfoundation/jeff-mlir.git
    GIT_TAG 7e11628de13d87798474386721c08f218b5f277e)
  function(_mqt_core_make_jeff_available)
    # Embed jeff and its dependencies in the compiler's consumers, like our own compiler libraries.
    set(BUILD_SHARED_LIBS OFF)
    # Cap'n Proto, which is fetched transitively by jeff-mlir, uses the generic BUILD_TESTING option
    # and defines a global `check` target when it is enabled. Do not let an embedding project's test
    # setting leak into this third-party dependency.
    set(BUILD_TESTING OFF)
    # jeff's transitive Cap'n Proto dependency contains source files that cannot share a unity
    # translation unit. Keep the complete dependency subtree out of unity builds.
    set(CMAKE_UNITY_BUILD OFF)
    FetchContent_MakeAvailable(jeff-mlir)
  endfunction()
  _mqt_core_make_jeff_available()
endif()

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

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
