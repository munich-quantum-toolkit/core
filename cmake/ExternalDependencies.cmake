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
  find_package(pybind11 2.13.6 CONFIG REQUIRED)
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
  set(GTEST_VERSION
      1.16.0
      CACHE STRING "Google Test version")
  set(GTEST_URL https://github.com/google/googletest/archive/refs/tags/v${GTEST_VERSION}.tar.gz)
  FetchContent_Declare(googletest URL ${GTEST_URL} FIND_PACKAGE_ARGS ${GTEST_VERSION} NAMES GTest)
  list(APPEND FETCH_PACKAGES googletest)
endif()

if(BUILD_MQT_CORE_CATALYST_PLUGIN)
  set(CATALYST_VERSION 0.12.0)
  # Check if the pennylane-catalyst package is installed in the python environment.
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m catalyst --version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE FOUND_CATALYST_VERSION)
  if(FOUND_CATALYST_VERSION)
    message(STATUS "Found pennylane-catalyst ${CATALYST_VERSION} in python environment.")
    # Check if the version is compatible.
    if(FOUND_CATALYST_VERSION VERSION_LESS ${CATALYST_VERSION})
      message(
        WARNING
          "pennylane-catalyst version ${FOUND_CATALYST_VERSION} in python environment is not compatible."
      )
    else()
      # Detect the installed catalyst include files.
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -m catalyst --include_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE CATALYST_INCLUDE_DIRS)
    endif()
  else()
    set(CATALYST_URL
        https://github.com/PennyLaneAI/catalyst/archive/refs/tags/v${CATALYST_VERSION}.tar.gz)
    FetchContent_Declare(catalyst URL ${CATALYST_URL} FIND_PACKAGE_ARGS ${CATALYST_VERSION})
    # todo: Find the include directory, generate header via TableGen if needed (at build time)
    # Catalyst is purposely not added to the list of fetch packages, because it is not meant to be
    # built or linked. We only need the include files. list(APPEND FETCH_PACKAGES catalyst)
  endif()

  if(NOT CATALYST_INCLUDE_DIRS)
    message(
      FATAL_ERROR
        "The include directory of the pennylane-catalyst package could not be retrieved. Please ensure that the catalyst is installed correctly."
    )
  endif()
endif()

# Make all declared dependencies available.
FetchContent_MakeAvailable(${FETCH_PACKAGES})
