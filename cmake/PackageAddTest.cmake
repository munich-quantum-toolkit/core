# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# macro to add a test executable for one of the project libraries
macro(PACKAGE_ADD_TEST testname linklibs)
  if(NOT TARGET ${testname})
    # create an executable in which the tests will be stored
    add_executable(${testname} ${ARGN})
    # link the Google test infrastructure and a default main function to the test executable.
    target_link_libraries(${testname} PRIVATE ${linklibs} gmock gtest_main MQT::ProjectOptions
                                              MQT::ProjectWarnings)
    # discover tests
    gtest_discover_tests(
      ${testname}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" DISCOVERY_TIMEOUT 60)
    set_target_properties(${testname} PROPERTIES FOLDER tests)
  endif()
endmacro()

macro(PACKAGE_ADD_TEST_WITH_WORKING_DIR testname linklibs test_working_directory)
  if(NOT TARGET ${testname})
    # create an executable in which the tests will be stored
    add_executable(${testname} ${ARGN})
    # link the Google test infrastructure and a default main function to the test executable.
    target_link_libraries(${testname} PRIVATE ${linklibs} gmock gtest_main MQT::ProjectOptions
                                              MQT::ProjectWarnings)
    # discover tests
    gtest_discover_tests(
      ${testname}
      WORKING_DIRECTORY ${test_working_directory}
      PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${test_working_directory}" DISCOVERY_TIMEOUT 60)
    set_target_properties(${testname} PROPERTIES FOLDER tests)
  endif()
endmacro()
