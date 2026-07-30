# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_library(mqt-core-qir-test-utils STATIC test_utils.cpp)
target_link_libraries(mqt-core-qir-test-utils PUBLIC gtest)
target_include_directories(mqt-core-qir-test-utils PUBLIC ${PROJECT_SOURCE_DIR}/test)
target_compile_definitions(mqt-core-qir-test-utils
                           PRIVATE QIR_FILES_DIR="${PROJECT_SOURCE_DIR}/test/circuits")
