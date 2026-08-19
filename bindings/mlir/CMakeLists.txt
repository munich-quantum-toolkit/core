# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME "${MQT_CORE_TARGET_NAME}-mlir-bindings")

if(NOT TARGET ${TARGET_NAME})
  # collect source files
  file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS *.cpp)

  # Qiskit's extension C API tables are translation-unit local. Each supported minor therefore gets
  # one source file with a private header snapshot.
  file(STRINGS qiskit/SupportedVersions.inc qiskit_version_registrations
       REGEX "^MQT_QISKIT_VERSION\\(")
  foreach(registration IN LISTS qiskit_version_registrations)
    string(
      REGEX
        MATCH
        "^MQT_QISKIT_VERSION\\(([0-9]+), *([0-9]+), *([0-9_]+), *([0-9]+), *([0-9]+\\.[0-9]+\\.[0-9]+), *\"[^\"]+\"\\)$"
        registration_match
        "${registration}")
    if(NOT registration_match)
      message(FATAL_ERROR "Invalid Qiskit version registration: ${registration}")
    endif()
    set(version_suffix "${CMAKE_MATCH_3}")
    set(version_release "${CMAKE_MATCH_5}")
    set(translation_source "${CMAKE_CURRENT_SOURCE_DIR}/qiskit/Qiskit${version_suffix}.cpp")
    if(NOT EXISTS "${translation_source}")
      message(
        FATAL_ERROR "Missing Qiskit ${version_release} translation source: ${translation_source}")
    endif()
    set(version_include "${PROJECT_SOURCE_DIR}/vendor/qiskit-c-api/${version_release}/include")
    if(NOT IS_DIRECTORY "${version_include}")
      message(FATAL_ERROR "Missing Qiskit ${version_release} C API headers: ${version_include}")
    endif()
    set_source_files_properties(
      ${translation_source} PROPERTIES COMPILE_DEFINITIONS QISKIT_PYTHON_EXTENSION
                                       INCLUDE_DIRECTORIES "${version_include}")
  endforeach()

  set(MQT_QISKIT_CAPI_CANDIDATE_INCLUDE
      ""
      CACHE PATH "CI-only include directory for an exact non-shipping Qiskit version")
  set(MQT_QISKIT_CAPI_CANDIDATE_VERSION
      ""
      CACHE STRING "CI-only exact Qiskit development version")
  if(MQT_QISKIT_CAPI_CANDIDATE_INCLUDE OR MQT_QISKIT_CAPI_CANDIDATE_VERSION)
    if(NOT IS_DIRECTORY "${MQT_QISKIT_CAPI_CANDIDATE_INCLUDE}")
      message(FATAL_ERROR "MQT_QISKIT_CAPI_CANDIDATE_INCLUDE must be a directory")
    endif()
    if(NOT MQT_QISKIT_CAPI_CANDIDATE_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.[0-9]+[^;\"]*$")
      message(FATAL_ERROR "MQT_QISKIT_CAPI_CANDIDATE_VERSION must be one exact version")
    endif()
    set(QISKIT_MAJOR "${CMAKE_MATCH_1}")
    set(QISKIT_MINOR "${CMAKE_MATCH_2}")
    set(QISKIT_FACTORY createCandidateTranslation)
    set(QISKIT_EXACT_API 1)
    set(QISKIT_LABEL "candidate ${MQT_QISKIT_CAPI_CANDIDATE_VERSION}")
    set(candidate_source "${CMAKE_CURRENT_BINARY_DIR}/QiskitCandidate.cpp")
    configure_file(qiskit/QiskitMinor.cpp.in ${candidate_source} @ONLY)
    list(APPEND SOURCES ${candidate_source})
    set_source_files_properties(
      ${candidate_source}
      PROPERTIES COMPILE_DEFINITIONS QISKIT_PYTHON_EXTENSION
                 INCLUDE_DIRECTORIES
                 "${MQT_QISKIT_CAPI_CANDIDATE_INCLUDE};${CMAKE_CURRENT_SOURCE_DIR}/qiskit")
  endif()

  # declare the Python module
  add_mqt_python_binding(
    CORE
    ${TARGET_NAME}
    ${SOURCES}
    MODULE_NAME
    mlir
    INSTALL_DIR
    .
    LINK_LIBS
    MQTCompilerQDMIAdapter
    MQTCompilerPipeline
    MLIRMQTDialect
    MLIRMQTUtils
    MLIRQCTranslationSupport
    MLIRQCODDFunctionality)

  if(MQT_QISKIT_CAPI_CANDIDATE_VERSION)
    target_compile_definitions(
      ${TARGET_NAME}
      PRIVATE MQT_QISKIT_CAPI_CANDIDATE_VERSION="${MQT_QISKIT_CAPI_CANDIDATE_VERSION}")
  endif()

  # install the Python stub file in editable mode for better IDE support
  if(SKBUILD_STATE STREQUAL "editable")
    install(
      FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/mlir.pyi
      DESTINATION .
      COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
  endif()
endif()
