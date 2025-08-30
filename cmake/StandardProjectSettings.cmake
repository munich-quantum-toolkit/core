# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
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
set_property(GLOBAL PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY CXX_EXTENSIONS OFF)

# Generate compile_commands.json to make it easier to work with clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE BOOL "Export compile commands" FORCE)

set(CMAKE_VERIFY_INTERFACE_HEADER_SETS
    ON
    CACHE BOOL "Verify interface header sets" FORCE)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  add_compile_options(-fcolor-diagnostics)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fdiagnostics-color=always)
else()
  message(STATUS "No colored compiler diagnostic set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
endif()

option(DEPLOY "Configure for deployment")
if(DEFINED ENV{DEPLOY})
  set(DEPLOY
      $ENV{DEPLOY}
      CACHE BOOL "Use deployment configuration from environment" FORCE)
  message(STATUS "Setting deployment configuration to '${DEPLOY}' from environment")
elseif(DEFINED ENV{CI})
  set(DEPLOY
      ON
      CACHE BOOL "Set deployment configuration to ON for CI" FORCE)
  message(STATUS "Setting deployment configuration to '${DEPLOY}' for CI")
endif()

# set deployment specific options
if(DEPLOY)
  # set the macOS deployment target appropriately
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "11.0"
      CACHE STRING "" FORCE)
endif()

# try to enable inter-procedural optimization per default for Release builds outside of deployment
if(NOT DEPLOY AND CMAKE_BUILD_TYPE STREQUAL "Release")
  option(ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" ON)
else()
  option(ENABLE_IPO "Enable Interprocedural Optimization, aka Link Time Optimization (LTO)" OFF)
endif()
if(ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
  # enable inter-procedural optimization if it is supported
  if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION
        TRUE
        CACHE BOOL "Enable Interprocedural Optimization" FORCE)
  else()
    message(DEBUG "IPO is not supported: ${ipo_output}")
  endif()
endif()

# export all symbols by default on Windows
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS
    ON
    CACHE BOOL "Export all symbols on Windows")
