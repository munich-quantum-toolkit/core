# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(add_mqt_python_binding package_name target_name)
  # parse the arguments
  cmake_parse_arguments(ARG "" "MODULE_NAME;INSTALL_DIR" "LINK_LIBS" ${ARGN})
  set(SOURCES ${ARG_UNPARSED_ARGUMENTS})

  # declare the Python module
  pybind11_add_module(
    # name of the extension
    ${target_name}
    # Pprefer thin LTO if available
    THIN_LTO
    # optimize the bindings for size
    OPT_SIZE
    # source code goes here
    ${SOURCES})

  # set default "." for INSTALL_DIR
  if(NOT ARG_INSTALL_DIR)
    set(ARG_INSTALL_DIR ".")
  endif()

  if(ARG_MODULE_NAME)
    # the library name must be the same as the module name
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${ARG_MODULE_NAME})
    target_compile_definitions(${target_name}
                               PRIVATE MQT_${package_name}_MODULE_NAME=${ARG_MODULE_NAME})
  else()
    # use the target name as the module name
    target_compile_definitions(${target_name}
                               PRIVATE MQT_${package_name}_MODULE_NAME=${target_name})
  endif()

  # add project libraries to the link libraries
  list(APPEND ARG_LINK_LIBS MQT::ProjectOptions MQT::ProjectWarnings)

  # Set c++ standard
  target_compile_features(${target_name} PRIVATE cxx_std_20)

  # link the required libraries
  target_link_libraries(${target_name} PRIVATE ${ARG_LINK_LIBS})

  # install directive for scikit-build-core
  install(
    TARGETS ${target_name}
    DESTINATION ${ARG_INSTALL_DIR}
    COMPONENT ${MQT_${package_name}_TARGET_NAME}_Python)
endfunction()
