# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# add various subdirectories containing tests
add_subdirectory(algorithms)
add_subdirectory(circuit_optimizer)
add_subdirectory(dd)
add_subdirectory(ir)
add_subdirectory(na)
add_subdirectory(zx)
add_subdirectory(qir)
add_subdirectory(qdmi)
add_subdirectory(fomac)

set(MQT_CORE_PARENT_SPDLOG_TEST_ARGS
    -G
    "${CMAKE_GENERATOR}"
    -S
    "${PROJECT_SOURCE_DIR}/test/cmake/parent-spdlog"
    -B
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/parent-spdlog"
    "-DMQT_CORE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DCMAKE_BUILD_TYPE=$<CONFIG>"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DMLIR_DIR=${MLIR_DIR}")
foreach(dependency IN ITEMS jeff-mlir jeff capnproto nlohmann_json boost_mp qdmi)
  FetchContent_GetProperties("${dependency}" SOURCE_DIR dependency_source_dir)
  if(dependency_source_dir)
    string(TOUPPER "${dependency}" dependency_upper)
    list(APPEND MQT_CORE_PARENT_SPDLOG_TEST_ARGS
         "-DFETCHCONTENT_SOURCE_DIR_${dependency_upper}=${dependency_source_dir}")
  endif()
endforeach()
add_test(NAME mqt-core-cmake-parent-spdlog COMMAND ${CMAKE_COMMAND} --fresh
                                                   ${MQT_CORE_PARENT_SPDLOG_TEST_ARGS})
# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
