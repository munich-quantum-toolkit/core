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
  list(FILTER SOURCES EXCLUDE REGEX "/qiskit/Adapter[0-9]+\\.cpp$")

  # Qiskit's extension C-API tables are translation-unit local. Keep each supported minor in one
  # private object target and never expose its headers.
  file(STRINGS qiskit/SupportedAdapters.inc qiskit_adapter_registrations
       REGEX "^MQT_QISKIT_ADAPTER\\(")
  set(qiskit_adapter_objects)
  foreach(registration IN LISTS qiskit_adapter_registrations)
    string(
      REGEX
        MATCH
        "^MQT_QISKIT_ADAPTER\\(([0-9]+), *([0-9]+), *([0-9]+), *([0-9]+\\.[0-9]+\\.[0-9]+), *\"[^\"]+\"\\)$"
        registration_match
        "${registration}")
    if(NOT registration_match)
      message(FATAL_ERROR "Invalid Qiskit adapter registration: ${registration}")
    endif()
    set(adapter_suffix "${CMAKE_MATCH_3}")
    set(adapter_release "${CMAKE_MATCH_4}")
    set(adapter_target "${TARGET_NAME}-qiskit-${adapter_suffix}")
    add_library(${adapter_target} OBJECT qiskit/Adapter${adapter_suffix}.cpp)
    target_compile_features(${adapter_target} PRIVATE cxx_std_20)
    target_compile_definitions(${adapter_target} PRIVATE QISKIT_PYTHON_EXTENSION)
    target_include_directories(
      ${adapter_target} SYSTEM
      PRIVATE ${PROJECT_SOURCE_DIR}/vendor/qiskit-c-api/${adapter_release}/include)
    target_link_libraries(${adapter_target} PRIVATE MQT::ProjectOptions MQT::ProjectWarnings
                                                    Python::Module)
    set_target_properties(${adapter_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    list(APPEND qiskit_adapter_objects $<TARGET_OBJECTS:${adapter_target}>)
  endforeach()

  set(MQT_QISKIT_CAPI_CANDIDATE_INCLUDE
      ""
      CACHE PATH "CI-only include directory for an exact non-shipping Qiskit adapter")
  set(MQT_QISKIT_CAPI_CANDIDATE_VERSION
      ""
      CACHE STRING "CI-only exact Qiskit development version for the candidate adapter")
  if(MQT_QISKIT_CAPI_CANDIDATE_INCLUDE OR MQT_QISKIT_CAPI_CANDIDATE_VERSION)
    if(NOT IS_DIRECTORY "${MQT_QISKIT_CAPI_CANDIDATE_INCLUDE}")
      message(FATAL_ERROR "MQT_QISKIT_CAPI_CANDIDATE_INCLUDE must be a directory")
    endif()
    if(NOT MQT_QISKIT_CAPI_CANDIDATE_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.[0-9]+[^;\"]*$")
      message(FATAL_ERROR "MQT_QISKIT_CAPI_CANDIDATE_VERSION must be one exact version")
    endif()
    set(ADAPTER_MAJOR "${CMAKE_MATCH_1}")
    set(ADAPTER_MINOR "${CMAKE_MATCH_2}")
    set(ADAPTER_FACTORY createCandidateAdapter)
    set(ADAPTER_FINAL_ONLY 0)
    set(ADAPTER_EXACT_API 1)
    set(ADAPTER_LABEL "candidate ${MQT_QISKIT_CAPI_CANDIDATE_VERSION}")
    configure_file(qiskit/AdapterMinor.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/AdapterCandidate.cpp
                   @ONLY)
    set(candidate_target "${TARGET_NAME}-qiskit-candidate")
    add_library(${candidate_target} OBJECT ${CMAKE_CURRENT_BINARY_DIR}/AdapterCandidate.cpp)
    target_compile_features(${candidate_target} PRIVATE cxx_std_20)
    target_compile_definitions(${candidate_target} PRIVATE QISKIT_PYTHON_EXTENSION)
    target_include_directories(${candidate_target} SYSTEM
                               PRIVATE ${MQT_QISKIT_CAPI_CANDIDATE_INCLUDE})
    target_include_directories(${candidate_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/qiskit)
    target_link_libraries(${candidate_target} PRIVATE MQT::ProjectOptions MQT::ProjectWarnings
                                                      Python::Module)
    set_target_properties(${candidate_target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    list(APPEND qiskit_adapter_objects $<TARGET_OBJECTS:${candidate_target}>)
  endif()

  # declare the Python module
  add_mqt_python_binding_nanobind(
    CORE
    ${TARGET_NAME}
    ${SOURCES}
    MODULE_NAME
    mlir
    INSTALL_DIR
    .
    LINK_LIBS
    MQTCompilerFoMaCAdapter
    MQTCompilerPipeline
    MQT::CoreIR)

  target_sources(${TARGET_NAME} PRIVATE ${qiskit_adapter_objects})
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
