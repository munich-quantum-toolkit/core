# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
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
  if(NOT SKBUILD)
    # Manually detect the installed pybind11 package and import it into CMake.
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE pybind11_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
  endif()

  message(STATUS "Python executable: ${Python_EXECUTABLE}")

  # add pybind11 library
  find_package(pybind11 3.0.0 CONFIG REQUIRED)
endif()

set(JSON_VERSION
    3.12.0
    CACHE STRING "nlohmann_json version")
set(JSON_URL https://github.com/nlohmann/json/releases/download/v${JSON_VERSION}/json.tar.xz)
set(JSON_SystemInclude
    ON
    CACHE INTERNAL "Treat the library headers like system headers")
cmake_dependent_option(JSON_Install "Install nlohmann_json library" ON "MQT_CORE_INSTALL" OFF)
FetchContent_Declare(nlohmann_json URL ${JSON_URL} FIND_PACKAGE_ARGS ${JSON_VERSION})
list(APPEND FETCH_PACKAGES nlohmann_json)

option(USE_SYSTEM_BOOST "Whether to try to use the system Boost installation" OFF)
set(BOOST_MIN_VERSION
    1.80.0
    CACHE STRING "Minimum required Boost version")
if(USE_SYSTEM_BOOST)
  find_package(Boost ${BOOST_MIN_VERSION} CONFIG REQUIRED)
else()
  set(BOOST_MP_STANDALONE
      ON
      CACHE INTERNAL "Use standalone boost multiprecision")
  set(BOOST_VERSION
      1_86_0
      CACHE INTERNAL "Boost version")
  set(BOOST_URL
      https://github.com/boostorg/multiprecision/archive/refs/tags/Boost_${BOOST_VERSION}.tar.gz)
  FetchContent_Declare(boost_mp URL ${BOOST_URL} FIND_PACKAGE_ARGS ${BOOST_MIN_VERSION} CONFIG
                                    NAMES boost_multiprecision)
  list(APPEND FETCH_PACKAGES boost_mp)
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

set(Protobuf_VERSION
    31.1
    CACHE STRING "protobuf version")
set(Protobuf_URL
    https://github.com/protocolbuffers/protobuf/releases/download/v${Protobuf_VERSION}/protobuf-${Protobuf_VERSION}.tar.gz
)
# the default of the following is ON, they are just here to make more explicit that they are
# required
set(protobuf_BUILD_PROTOBUF_BINARIES ON)
set(protobuf_BUILD_PROTOC_BINARIES ON)
set(protobuf_BUILD_LIBUPB ON)
# the default of the following is ON, but we do not need the tests
set(protobuf_BUILD_TESTS OFF)
FetchContent_Declare(protobuf URL ${Protobuf_URL} FIND_PACKAGE_ARGS ${Protobuf_VERSION} CONFIG
                                  NAMES protobuf)
list(APPEND FETCH_PACKAGES protobuf)

# cmake-format: off
set(QDMI_VERSION 1.2.0
        CACHE STRING "QDMI version")
set(QDMI_REV "9034e653a6e368579d99e352ecd8390e6e947bc6"
        CACHE STRING "QDMI identifier (tag, branch or commit hash)")
set(QDMI_REPO_OWNER "Munich-Quantum-Software-Stack"
        CACHE STRING "QDMI repository owner (change when using a fork)")
# cmake-format: on
FetchContent_Declare(
  qdmi
  GIT_REPOSITORY https://github.com/${QDMI_REPO_OWNER}/qdmi.git
  GIT_TAG ${QDMI_REV}
  FIND_PACKAGE_ARGS ${QDMI_VERSION})
list(APPEND FETCH_PACKAGES qdmi)

set(SPDLOG_VERSION
    1.15.3
    CACHE STRING "spdlog version")
set(SPDLOG_URL https://github.com/gabime/spdlog/archive/refs/tags/v${SPDLOG_VERSION}.tar.gz)
# Add position independent code for spdlog, this is required for python bindings on linux
set(SPDLOG_BUILD_PIC ON)
FetchContent_Declare(spdlog URL ${SPDLOG_URL} FIND_PACKAGE_ARGS ${SPDLOG_VERSION})
list(APPEND FETCH_PACKAGES spdlog)

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
