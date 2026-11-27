# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

find_program(LLC_EXECUTABLE NAMES llc HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)

set(runtime_test mqt-core-qir-runtime-test)
package_add_test(${runtime_test} MQT::CoreQIRRuntime test_qir_runtime.cpp)
mqt_llvm_target_disable_rtti(${runtime_test})

file(GLOB qir_files CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/test/circuits/*.ll")
foreach(source IN LISTS qir_files)
  get_filename_component(circuit_stem "${source}" NAME_WE)
  string(REGEX REPLACE "([A-Z])" "-\\1" target "${circuit_stem}")
  string(TOLOWER "mqt-core-qir${target}-ll" target)
  add_custom_command(
    OUTPUT "${circuit_stem}.o"
    COMMAND
      "${LLC_EXECUTABLE}" -filetype=obj -relocation-model=pic "${source}" -o "${circuit_stem}.o"
    DEPENDS "${source}"
    VERBATIM)
  add_executable(${target} "${circuit_stem}.o")
  set_target_properties(${target} PROPERTIES BUILD_WITH_INSTALL_RPATH FALSE LINKER_LANGUAGE CXX)
  target_link_libraries(${target} PRIVATE MQT::CoreQIRRuntime)
  add_test(NAME ${target} COMMAND ${target})
endforeach()
