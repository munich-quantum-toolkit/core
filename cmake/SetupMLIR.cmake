# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# set the include directory for the build tree
set(MQT_MLIR_SOURCE_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/mlir/include")
set(MQT_MLIR_BUILD_INCLUDE_DIR "${PROJECT_BINARY_DIR}/mlir/include")
set(MQT_MLIR_MIN_VERSION "23.1" CACHE STRING "Required MLIR release series")
option(MQT_CORE_AUTO_INSTALL_MLIR "Download LLVM/MLIR when no compatible SDK is found" ON)
set(MQT_CORE_MLIR_VERSION "23.1.2" CACHE STRING "LLVM/MLIR version to download")
get_property(_mqt_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _mqt_build_type)
set(_mqt_mlir_assertions_default OFF)
if(NOT _mqt_multi_config AND _mqt_build_type MATCHES "^(DEBUG|RELWITHDEBINFO)$")
  set(_mqt_mlir_assertions_default ON)
endif()
option(MQT_CORE_MLIR_ASSERTIONS "Use assertions in the downloaded LLVM/MLIR SDK"
       ${_mqt_mlir_assertions_default})

# Attempt to load MLIR_DIR from a local .env file for developer convenience.
if(NOT DEFINED MLIR_DIR AND EXISTS "${PROJECT_SOURCE_DIR}/.env")
  file(STRINGS "${PROJECT_SOURCE_DIR}/.env" MQT_CORE_DOTENV_LINES)
  foreach(MQT_CORE_DOTENV_LINE IN LISTS MQT_CORE_DOTENV_LINES)
    if(MQT_CORE_DOTENV_LINE MATCHES
       "^[ \t]*(export[ \t]+)?MLIR_DIR[ \t]*=[ \t]*['\"]?([^'\"]+)['\"]?[ \t]*$")
      file(TO_CMAKE_PATH "${CMAKE_MATCH_2}" MQT_CORE_DOTENV_MLIR_DIR)
      set(MLIR_DIR "${MQT_CORE_DOTENV_MLIR_DIR}" CACHE PATH "Path to MLIRConfig.cmake" FORCE)
      message(STATUS "Using MLIR_DIR from ${PROJECT_SOURCE_DIR}/.env: ${MLIR_DIR}")
      break()
    endif()
  endforeach()
endif()

# Imported targets cannot be replaced by another SDK during configuration.
if(TARGET LLVMSupport OR TARGET MLIRSupport)
  if(NOT LLVM_PACKAGE_VERSION
     OR NOT "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}" VERSION_EQUAL MQT_MLIR_MIN_VERSION)
    message(FATAL_ERROR "The embedding project must provide a compatible LLVM/MLIR SDK.")
  endif()
  find_package(MLIR ${LLVM_PACKAGE_VERSION} EXACT CONFIG REQUIRED)
else()
  find_package(MLIR ${MQT_MLIR_MIN_VERSION} CONFIG QUIET)
endif()
if(NOT MLIR_FOUND)
  if(NOT MQT_CORE_MLIR_VERSION MATCHES "^([0-9]+\\.[0-9]+)\\.[0-9]+$"
     OR NOT CMAKE_MATCH_1 VERSION_EQUAL MQT_MLIR_MIN_VERSION)
    message(FATAL_ERROR "MQT_CORE_MLIR_VERSION must be a ${MQT_MLIR_MIN_VERSION}.x release.")
  endif()
  if(MQT_CORE_MLIR_ASSERTIONS)
    set(_mqt_mlir_assertions ON)
  else()
    set(_mqt_mlir_assertions OFF)
  endif()
  set(_mqt_mlir_prefix
      "${PROJECT_BINARY_DIR}/_deps/llvm-${MQT_CORE_MLIR_VERSION}-${_mqt_mlir_assertions}")

  # shortcut: setup-mlir selects the host SDK; target selection needs upstream cross-build support.
  if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "Automatic LLVM/MLIR setup supports native builds only. Set MLIR_DIR.")
  endif()

  if(NOT EXISTS "${_mqt_mlir_prefix}/.complete")
    if(NOT MQT_CORE_AUTO_INSTALL_MLIR OR FETCHCONTENT_FULLY_DISCONNECTED)
      message(
        FATAL_ERROR "No compatible LLVM/MLIR SDK found and downloading is disabled. Set MLIR_DIR.")
    endif()
    include(FetchContent)
    # Keep the installer and its version manifest at the same revision.
    fetchcontent_declare(
      setup_mlir
      URL
        https://github.com/munich-quantum-software/setup-mlir/archive/8d3eae73d0f0196fd30c787d91ebd2018cb6c709.tar.gz
      URL_HASH SHA256=cc4d195d0ab5404bf218d7f225d8fea2510106c002773badef6b1d98a44aedb6)
    fetchcontent_makeavailable(setup_mlir)
    if(WIN32)
      set(_mqt_mlir_command
          pwsh
          -NoProfile
          -ExecutionPolicy
          Bypass
          -File
          "${setup_mlir_SOURCE_DIR}/installation/setup-mlir.ps1"
          -llvm_version
          "${MQT_CORE_MLIR_VERSION}"
          -install_prefix
          "${_mqt_mlir_prefix}")
      if(NOT MQT_CORE_MLIR_ASSERTIONS)
        list(APPEND _mqt_mlir_command -no_assertions)
      endif()
    else()
      set(_mqt_mlir_command
          bash
          "${setup_mlir_SOURCE_DIR}/installation/setup-mlir.sh"
          -v
          "${MQT_CORE_MLIR_VERSION}"
          -p
          "${_mqt_mlir_prefix}"
          -a
          "${_mqt_mlir_assertions}")
    endif()
    message(
      STATUS
        "Installing LLVM/MLIR ${MQT_CORE_MLIR_VERSION} (assertions ${_mqt_mlir_assertions}) in ${_mqt_mlir_prefix}")
    execute_process(COMMAND ${_mqt_mlir_command} COMMAND_ERROR_IS_FATAL ANY)
    file(WRITE "${_mqt_mlir_prefix}/.complete" "")
  endif()

  # Do not cache these paths: changing the requested SDK must select a new prefix.
  set(LLVM_DIR "${_mqt_mlir_prefix}/lib/cmake/llvm")
  set(MLIR_DIR "${_mqt_mlir_prefix}/lib/cmake/mlir")
  find_package(MLIR ${MQT_CORE_MLIR_VERSION} EXACT CONFIG REQUIRED)
endif()
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM/MLIR ${MLIR_VERSION}, assertions ${LLVM_ENABLE_ASSERTIONS}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Keep direct LLVM consumers compatible with an RTTI-free LLVM installation. Ordinary MLIR targets
# receive this policy through llvm_update_compile_flags.
function(mqt_llvm_target_disable_rtti target_name)
  if(NOT TARGET ${target_name})
    message(FATAL_ERROR "Cannot configure RTTI for missing target ${target_name}.")
  endif()
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(
      ${target_name}
      PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${LLVM_CXXFLAGS_RTTI_DISABLE}>)
  endif()
endfunction()

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
include_directories(${MQT_MLIR_SOURCE_INCLUDE_DIR})
include_directories(${MQT_MLIR_BUILD_INCLUDE_DIR})
separate_arguments(MQT_LLVM_DEFINITIONS NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${MQT_LLVM_DEFINITIONS})

# Generate MLIR artifacts, including documentation, in the build tree.
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
