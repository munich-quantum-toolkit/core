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

# Give clang-tidy compile settings for the standalone installed-wheel consumer.
if(TARGET MQT::CoreQDMI AND TARGET MQT::CoreDD)
  find_package(Threads REQUIRED)
  add_library(mqt-core-wheel-consumer-lint OBJECT EXCLUDE_FROM_ALL release/consumer/main.cpp)
  target_link_libraries(mqt-core-wheel-consumer-lint PRIVATE MQT::CoreQDMI MQT::CoreDD
                                                             Threads::Threads)
endif()

# copy test circuits to build directory
file(COPY ${PROJECT_SOURCE_DIR}/test/circuits DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
