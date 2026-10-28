# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME "${MQT_CORE_TARGET_NAME}-bindings")

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)

# Qiskit's extension C API tables are translation-unit local. Each supported minor therefore gets
# one source file with a private header snapshot.
file(
  STRINGS Target/Qiskit/SupportedVersions.inc qiskit_version_registrations
  REGEX "^MQT_QISKIT_VERSION\\(")
foreach(registration IN LISTS qiskit_version_registrations)
  string(REGEX MATCH
         "^MQT_QISKIT_VERSION\\(([0-9]+), *([0-9]+), *(2\\.[0-9]+\\.[0-9]+), *\"[^\"]+\"\\)$"
         registration_match "${registration}")
  if(NOT registration_match)
    message(FATAL_ERROR "Invalid Qiskit version registration: ${registration}")
  endif()
  set(version_minor "${CMAKE_MATCH_1}")
  set(version_release "${CMAKE_MATCH_3}")
  set(translation_source "${CMAKE_CURRENT_SOURCE_DIR}/Target/Qiskit/Qiskit2_${version_minor}.cpp")
  if(NOT EXISTS "${translation_source}")
    message(
      FATAL_ERROR "Missing Qiskit ${version_release} translation source: ${translation_source}")
  endif()
  set(version_include "${PROJECT_SOURCE_DIR}/vendor/qiskit-c-api/${version_release}/include")
  if(NOT IS_DIRECTORY "${version_include}")
    message(FATAL_ERROR "Missing Qiskit ${version_release} C API headers: ${version_include}")
  endif()
  set_source_files_properties(
    ${translation_source}
    TARGET_DIRECTORY ${TARGET_NAME}
    PROPERTIES COMPILE_DEFINITIONS QISKIT_PYTHON_EXTENSION INCLUDE_DIRECTORIES "${version_include}")
endforeach()

target_sources(${TARGET_NAME} PRIVATE ${SOURCES})
target_link_libraries(
  ${TARGET_NAME}
  PRIVATE
    MQT::CoreBenchGenerate
    MQTCompilerQDMIAdapter
    MQTCompilerPipeline
    LLVMIRReader
    MLIRAnalysis
    MLIRMQTDialect
    MLIRMQTUtils
    MLIRQCTranslationSupport
    MLIRQCODDFunctionality)

# install the Python stub file in editable mode for better IDE support
if(MQT_CORE_INSTALL AND SKBUILD_STATE STREQUAL "editable")
  install(
    FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/mlir.pyi
    DESTINATION .
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()
