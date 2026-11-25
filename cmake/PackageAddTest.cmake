# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Keep individual GoogleTest results while amortizing process startup across a suite.
function(mqt_add_gtest target shards)
  if(NOT shards)
    set(shards 1)
  endif()
  math(EXPR last_shard "${shards} - 1")
  foreach(shard RANGE ${last_shard})
    set(name ${target})
    if(shards GREATER 1)
      string(APPEND name "-${shard}")
    endif()
    add_test(
      NAME ${name}
      COMMAND ${target} "--gtest_output=xml:${CMAKE_CURRENT_BINARY_DIR}/${name}.xml")
    set_tests_properties(${name} PROPERTIES COST ${shards} ${ARGN})
    if(shards GREATER 1)
      set_tests_properties(
        ${name}
        PROPERTIES ENVIRONMENT "GTEST_TOTAL_SHARDS=${shards};GTEST_SHARD_INDEX=${shard}")
    endif()
  endforeach()
endfunction()

function(package_add_test testname linklibs)
  if(TARGET ${testname})
    return()
  endif()
  cmake_parse_arguments(PARSE_ARGV 2 arg "ISOLATED" "WORKING_DIRECTORY;SHARDS" "")
  if(NOT arg_WORKING_DIRECTORY)
    set(arg_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  add_executable(${testname} ${arg_UNPARSED_ARGUMENTS})
  set_target_properties(
    ${testname}
    PROPERTIES
      FOLDER tests BUILD_WITH_INSTALL_RPATH FALSE VS_DEBUGGER_WORKING_DIRECTORY
      "${arg_WORKING_DIRECTORY}")
  target_link_libraries(
    ${testname}
    PRIVATE
      ${linklibs} MQTCoreTestSupport GTest::gmock GTest::gtest_main MQT::ProjectOptions
      MQT::ProjectWarnings)
  target_compile_features(${testname} PRIVATE cxx_std_20)
  if(arg_ISOLATED)
    gtest_discover_tests(
      ${testname}
      DISCOVERY_MODE PRE_TEST
      WORKING_DIRECTORY
        "${CMAKE_CURRENT_BINARY_DIR}"
        # Discover beside the executable, then run cases in their fixture directory.
      PROPERTIES "$<1:WORKING_DIRECTORY>" "${arg_WORKING_DIRECTORY}"
      DISCOVERY_TIMEOUT 60)
  else()
    mqt_add_gtest(${testname} "${arg_SHARDS}" WORKING_DIRECTORY "${arg_WORKING_DIRECTORY}")
  endif()
endfunction()
