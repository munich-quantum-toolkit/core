# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(add_mqt_python_binding package_name target_name)
  cmake_parse_arguments(ARG "" "MODULE_NAME;INSTALL_DIR" "LINK_LIBS" ${ARGN})
  set(SOURCES ${ARG_UNPARSED_ARGUMENTS})

  # nanobind 3.0 ignores scikit-build-core's abi3t signal on Windows. NB_ABI configures the nested
  # nanobind_add_module call.
  if(WIN32 AND Py_TARGET_ABI3T)
    set(NB_ABI "${Python_VERSION_MAJOR}${Python_VERSION_MINOR}t")
  endif()

  nanobind_add_module(
    # Name of the extension
    ${target_name}
    # Enable free-threaded support
    FREE_THREADED
    # Suppress compiler warnings from the nanobind library
    NB_SUPPRESS_WARNINGS
    # Use nanobind's shared runtime. Split mode enables the Stable ABI internally.
    BACKEND_MODULE
    nanobind_backend
    # Source files
    ${SOURCES})

  # Set C++ standard
  target_compile_features(${target_name} PRIVATE cxx_std_20)

  if(ARG_MODULE_NAME)
    # The library name must be the same as the module name
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${ARG_MODULE_NAME})
    target_compile_definitions(${target_name}
                               PRIVATE MQT_${package_name}_MODULE_NAME=${ARG_MODULE_NAME})
    set(module_name ${ARG_MODULE_NAME})
  else()
    # Use the target name as the module name
    target_compile_definitions(${target_name}
                               PRIVATE MQT_${package_name}_MODULE_NAME=${target_name})
    set(module_name ${target_name})
  endif()

  # Keep statically linked dependencies local.
  if(APPLE)
    target_link_options(${target_name} PRIVATE "LINKER:-exported_symbol,_PyInit_${module_name}")
  elseif(UNIX)
    target_link_options(${target_name} PRIVATE "LINKER:--exclude-libs,ALL")

    # nanobind 3.0 omits section garbage collection from split-mode targets
    target_link_options(
      ${target_name}
      PRIVATE
      "$<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>,$<CONFIG:RelWithDebInfo>>:LINKER:--gc-sections>"
    )
  elseif(WIN32)
    set_target_properties(${target_name} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS OFF)
  endif()

  # Add project libraries to the link libraries
  list(APPEND ARG_LINK_LIBS MQT::ProjectOptions MQT::ProjectWarnings)

  target_link_libraries(${target_name} PRIVATE ${ARG_LINK_LIBS})

  # Set default "." for INSTALL_DIR
  if(NOT ARG_INSTALL_DIR)
    set(ARG_INSTALL_DIR ".")
  endif()

  # Install directive for scikit-build-core
  install(
    TARGETS ${target_name}
    DESTINATION ${ARG_INSTALL_DIR}
    COMPONENT ${MQT_${package_name}_TARGET_NAME}_Python)
endfunction()
