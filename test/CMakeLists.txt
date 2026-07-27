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

if(MLIR_FOUND)
  set(MQT_CORE_QDMI_CONSUMER_SOURCE_DIR "${PROJECT_SOURCE_DIR}/test/cmake/qdmi-consumer")
  set(MQT_CORE_QDMI_CONSUMER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/cmake/qdmi-consumer")
  set(MQT_CORE_QDMI_CONSUMER_ARGS
      -G
      "${CMAKE_GENERATOR}"
      -S
      "${MQT_CORE_QDMI_CONSUMER_SOURCE_DIR}"
      -B
      "${MQT_CORE_QDMI_CONSUMER_BINARY_DIR}"
      "-DMQT_CORE_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
      "-DPARENT_SPDLOG_SOURCE_DIR=${spdlog_SOURCE_DIR}"
      "-DPARENT_SPDLOG_DIR=${spdlog_DIR}"
      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DMLIR_DIR=${MLIR_DIR}")
  foreach(dependency IN ITEMS nlohmann_json boost_mp qdmi)
    set(dependency_source_dir_variable "${dependency}_SOURCE_DIR")
    if(DEFINED ${dependency_source_dir_variable})
      string(TOUPPER "${dependency}" dependency_upper)
      list(APPEND MQT_CORE_QDMI_CONSUMER_ARGS
           "-DFETCHCONTENT_SOURCE_DIR_${dependency_upper}=${${dependency_source_dir_variable}}")
    endif()
  endforeach()

  add_test(NAME mqt-core-cmake-qdmi-consumer-configure COMMAND ${CMAKE_COMMAND} --fresh
                                                               ${MQT_CORE_QDMI_CONSUMER_ARGS})
  set_tests_properties(mqt-core-cmake-qdmi-consumer-configure
                       PROPERTIES FIXTURES_SETUP mqt-core-cmake-qdmi-consumer)

  add_test(NAME mqt-core-cmake-qdmi-consumer-build
           COMMAND ${CMAKE_COMMAND} --build "${MQT_CORE_QDMI_CONSUMER_BINARY_DIR}" --target
                   mqt-core-qdmi-consumer)
  set_tests_properties(mqt-core-cmake-qdmi-consumer-build PROPERTIES FIXTURES_REQUIRED
                                                                     mqt-core-cmake-qdmi-consumer)
endif()

# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
