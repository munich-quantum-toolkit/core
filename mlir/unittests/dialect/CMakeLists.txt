# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(testname "mqt-core-mlir-wireiterator-test")
file(GLOB_RECURSE WIREITERATOR_TEST_SOURCES *.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

if(NOT TARGET ${testname})
  # create an executable in which the tests will be stored
  add_executable(${testname} ${WIREITERATOR_TEST_SOURCES})
  # link the Google test infrastructure and a default main function to the test executable.
  target_link_libraries(${testname} PRIVATE GTest::gmock GTest::gtest_main ${dialect_libs})
  # discover tests
  gtest_discover_tests(${testname} DISCOVERY_TIMEOUT 60)
  set_target_properties(${testname} PROPERTIES FOLDER unittests)
endif()
