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
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
endif()

# Fetch jeff-mlir
fetchcontent_declare(
  jeff-mlir
  GIT_REPOSITORY https://github.com/unitaryfoundation/jeff-mlir.git
  GIT_TAG ab4d73a590d097c2b67b6c6c489444d9070fe4ab
  EXCLUDE_FROM_ALL)
block()
  # Cap'n Proto defines a global `check` target when BUILD_TESTING is enabled.
  # Keep the embedding project's test setting out of this dependency.
  set(BUILD_TESTING OFF)
  # jeff's transitive Cap'n Proto dependency contains source files that cannot share a unity
  # translation unit. Keep the complete dependency subtree out of unity builds.
  set(CMAKE_UNITY_BUILD OFF)
  # Cap'n Proto requires compiler extensions.
  set(CMAKE_CXX_EXTENSIONS ON)
  fetchcontent_makeavailable(jeff-mlir)
endblock()
if(MSVC AND TARGET capnp)
  # shortcut: Cap'n Proto has GCC-only pragmas; remove when guarded upstream.
  target_compile_options(capnp PRIVATE /wd4068)
endif()

if(BUILD_MQT_CORE_TESTS)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  # Disable the install instructions for GTest, as we do not need them.
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(GTEST_VERSION 1.17.0 CACHE STRING "Google Test version")
  set(GTEST_URL https://github.com/google/googletest/archive/refs/tags/v${GTEST_VERSION}.tar.gz)
  fetchcontent_declare(googletest URL ${GTEST_URL} FIND_PACKAGE_ARGS ${GTEST_VERSION} NAMES GTest)
  list(APPEND FETCH_PACKAGES googletest)
endif()

set(QDMI_MINIMUM_VERSION 1.4.0 CACHE STRING "Minimum QDMI version")
set(QDMI_VERSION 1.4.0 CACHE STRING "QDMI version")
set(QDMI_REV
    "6db03d83ad602aaa9fe928b5c3fc5cce84b10061" # v1.4.0
    CACHE STRING "QDMI identifier (tag, branch or commit hash)")
set(QDMI_REPO_OWNER "Munich-Quantum-Software-Stack"
    CACHE STRING "QDMI repository owner (change when using a fork)")
cmake_dependent_option(INSTALL_QDMI "Install QDMI library" ON "MQT_CORE_INSTALL" OFF)
fetchcontent_declare(
  qdmi
  GIT_REPOSITORY https://github.com/${QDMI_REPO_OWNER}/qdmi.git
  GIT_TAG ${QDMI_REV}
  FIND_PACKAGE_ARGS ${QDMI_MINIMUM_VERSION})
list(APPEND FETCH_PACKAGES qdmi)

fetchcontent_makeavailable(${FETCH_PACKAGES})
