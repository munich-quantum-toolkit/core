# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# enable organization of targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui, ccmake
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel"
                                               "RelWithDebInfo")
endif()

# Require C++ standard
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generate compile_commands.json to make it easier to work with clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "Export compile commands" FORCE)

set(CMAKE_VERIFY_INTERFACE_HEADER_SETS
    OFF
    CACHE BOOL "Verify interface header sets")

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  add_compile_options(-fcolor-diagnostics)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fdiagnostics-color=always)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/diagnostics:color)
else()
  message(STATUS "No colored compiler diagnostic set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
endif()

option(DEPLOY "Configure for deployment")
if(DEFINED ENV{DEPLOY})
  set(DEPLOY
      $ENV{DEPLOY}
      CACHE BOOL "Use deployment configuration from environment" FORCE)
  message(STATUS "Setting deployment configuration to '${DEPLOY}' from environment")
endif()

# set deployment specific options
if(DEPLOY)
  # set the macOS deployment target appropriately
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "13.3"
      CACHE STRING "" FORCE)
endif()

# MSVC /GL libraries repeat code generation at every test executable's link.
if(NOT DEPLOY
   AND CMAKE_BUILD_TYPE STREQUAL "Release"
   AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND BUILD_MQT_CORE_TESTS))
  option(ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" ON)
else()
  option(ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
endif()
set(ipo_supported FALSE)
if(ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
  if(NOT ipo_supported)
    message(DEBUG "IPO is not supported: ${ipo_output}")
  endif()
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION
    ${ipo_supported}
    CACHE BOOL "Enable Interprocedural Optimization" FORCE)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
