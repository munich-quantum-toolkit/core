# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# add various subdirectories containing tests
add_subdirectory(bench)
add_subdirectory(dd)
add_subdirectory(qdmi)

add_executable(mqt-core-dd-consumer cmake/installed_consumer/dd_consumer.cpp)
set_property(TARGET mqt-core-dd-consumer PROPERTY BUILD_WITH_INSTALL_RPATH FALSE)
target_link_libraries(mqt-core-dd-consumer PRIVATE MQT::CoreDD)
add_test(NAME mqt-core-dd-consumer COMMAND mqt-core-dd-consumer)
set_tests_properties(
  mqt-core-dd-consumer PROPERTIES ENVIRONMENT_MODIFICATION
                                  "PATH=path_list_prepend:$<TARGET_FILE_DIR:MQT::CoreDD>")

# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
