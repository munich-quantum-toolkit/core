# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# macro to add a test executable for one qir circuit
macro(ADD_QIR_CIRCUIT target_name circuit_path)
  if(NOT TARGET ${target_name})
    # Add a custom command to compile the .ll file to .o
    get_filename_component(circuit_name ${circuit_path} NAME_WE)
    add_custom_command(
      OUTPUT ${circuit_name}.o
      COMMAND clang -c ${circuit_path} -o ${circuit_name}.o
      DEPENDS ${circuit_path}
      COMMENT "Compiling ${circuit_path} to ${circuit_name}.o")
    add_executable(${target_name} ${circuit_name}.o)
    target_link_libraries(${target_name} PRIVATE MQT::CoreQIRRuntime)
  endif()
endmacro()

# function to convert camel case to dash case
function(camel_to_dash_lowercase input output)
  string(REGEX REPLACE "([A-Z])" "-\\1" result "${input}")
  string(TOLOWER "${result}" result)
  string(REGEX REPLACE "^-+" "" result "${result}")
  set(${output}
      "${result}"
      PARENT_SCOPE)
endfunction()

set(TARGET_NAME mqt-core-qir-runtime-test)

if(TARGET MQT::CoreQIRRuntime)
  # add test executable
  package_add_test(${TARGET_NAME} MQT::CoreQIRRuntime test_qir_runtime.cpp)

  # add tests for QIR files
  set(QIR_EXECUTABLES "")
  set(QIR_FILES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

  file(GLOB QIR_FILES "${QIR_FILES_DIR}/*.ll")
  foreach(QIR_EXAMPLE ${QIR_FILES})
    get_filename_component(QIR_EXAMPLE_NAME ${QIR_EXAMPLE} NAME_WE)
    camel_to_dash_lowercase(${QIR_EXAMPLE_NAME} QIR_EXAMPLE_NAME)
    add_qir_circuit(mqt-core-qir-${QIR_EXAMPLE_NAME}-ll ${QIR_EXAMPLE})
    list(APPEND QIR_EXECUTABLES "$<TARGET_FILE:mqt-core-qir-${QIR_EXAMPLE_NAME}-ll>")
    add_dependencies(${TARGET_NAME} mqt-core-qir-${QIR_EXAMPLE_NAME}-ll)
  endforeach()

  # transform QIR_EXECUTABLES to comma separated list of string ("...")
  string(REPLACE ";" "\",\"" QIR_EXECUTABLES "\"${QIR_EXECUTABLES}\"")

  # add all executables as initializer list of std::filesystem::path's
  target_compile_definitions(${TARGET_NAME} PRIVATE TEST_EXECUTABLES=${QIR_EXECUTABLES})
endif()
