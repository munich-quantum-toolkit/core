# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

option(MQT_CORE_REQUIRE_SDK_MOLD "Require mold from the LLVM SDK" OFF)

function(mqt_select_sdk_mold)
  if(NOT PROJECT_IS_TOP_LEVEL
     OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
     OR CMAKE_VERSION VERSION_LESS 3.29 OR CMAKE_CROSSCOMPILING)
    return()
  endif()
  if(DEFINED CMAKE_LINKER_TYPE
     OR LLVM_USE_LINKER
     OR LLVM_ENABLE_LTO OR NOT CMAKE_C_COMPILER_ID STREQUAL CMAKE_CXX_COMPILER_ID)
    return()
  endif()

  # Honor a linker selected through compiler driver flags in any active configuration.
  set(driver_flags "${CMAKE_C_COMPILER_ARG1} ${CMAKE_CXX_COMPILER_ARG1}")
  set(flag_vars CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  foreach(kind IN ITEMS EXE SHARED MODULE)
    list(APPEND flag_vars CMAKE_${kind}_LINKER_FLAGS)
  endforeach()
  foreach(flags IN LISTS flag_vars)
    string(APPEND driver_flags " ${${flags}}")
    foreach(config IN ITEMS ${CMAKE_BUILD_TYPE} ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER "${config}" config)
      string(APPEND driver_flags " ${${flags}_${config}}")
    endforeach()
  endforeach()
  if(driver_flags MATCHES "(^|[ 	])(-fuse-ld|--ld-path)(=|[ 	])")
    return()
  endif()

  set(mold "${LLVM_TOOLS_BINARY_DIR}/ld.mold")
  if(NOT EXISTS "${mold}")
    return()
  endif()
  # mold 3 protects shared-library exports during safe identical code folding.
  execute_process(COMMAND "${mold}" --version OUTPUT_VARIABLE mold_version ERROR_QUIET)
  if(NOT mold_version MATCHES "^mold ([0-9.]+)" OR CMAKE_MATCH_1 VERSION_LESS 3)
    return()
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(linker_flags "--ld-path=\"${mold}\"")
    if(ENABLE_IPO)
      set(llvmgold "${LLVM_LIBRARY_DIR}/LLVMgold.so")
      if(NOT EXISTS "${llvmgold}"
         OR NOT CMAKE_C_COMPILER_VERSION VERSION_EQUAL LLVM_PACKAGE_VERSION
         OR NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL LLVM_PACKAGE_VERSION)
        return()
      endif()
      list(APPEND linker_flags -Xlinker -plugin -Xlinker "\"${llvmgold}\"")
    endif()
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND EXISTS "${LLVM_INSTALL_PREFIX}/libexec/mold/ld")
    # Use only the SDK's linker search directory. GCC supplies its own LTO plugin.
    set(linker_flags "-B\"${LLVM_INSTALL_PREFIX}/libexec/mold/\"")
  else()
    return()
  endif()

  set(CMAKE_LINKER_TYPE mqt_sdk_mold PARENT_SCOPE)
  foreach(language IN ITEMS C CXX)
    set(CMAKE_${language}_USING_LINKER_mqt_sdk_mold "${linker_flags}" PARENT_SCOPE)
    # The separate IPO probe needs the same linker path and plugin as the real build.
    list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES CMAKE_${language}_USING_LINKER_mqt_sdk_mold)
  endforeach()
  set(CMAKE_TRY_COMPILE_PLATFORM_VARIABLES "${CMAKE_TRY_COMPILE_PLATFORM_VARIABLES}" PARENT_SCOPE)
endfunction()

mqt_select_sdk_mold()
if(MQT_CORE_REQUIRE_SDK_MOLD AND NOT CMAKE_LINKER_TYPE STREQUAL "mqt_sdk_mold")
  message(
    FATAL_ERROR
      "SDK mold is unavailable. Use a compatible LLVM SDK and compiler with no linker override.")
endif()

set(ipo_supported FALSE)
if(ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
  if(NOT ipo_supported AND CMAKE_LINKER_TYPE STREQUAL "mqt_sdk_mold")
    if(MQT_CORE_REQUIRE_SDK_MOLD)
      message(FATAL_ERROR "IPO with the SDK's mold failed:\n${ipo_output}")
    endif()
    message(STATUS "Falling back to the default linker after the SDK mold IPO check failed")
    unset(CMAKE_LINKER_TYPE)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
  endif()
  if(NOT ipo_supported)
    message(DEBUG "IPO is not supported: ${ipo_output}")
  endif()
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ${ipo_supported}
    CACHE BOOL "Enable Interprocedural Optimization" FORCE)
