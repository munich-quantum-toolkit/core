# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# The installed wheel and build-tree tests use different runtime directories.
function(mqt_study_test_runtime directory)
  get_property(
    targets
    DIRECTORY "${directory}"
    PROPERTY BUILDSYSTEM_TARGETS)
  foreach(target IN LISTS targets)
    get_target_property(kind "${target}" TYPE)
    if("${kind}" STREQUAL "EXECUTABLE" AND "${target}" MATCHES "^mqt.*test")
      set_property(TARGET "${target}" PROPERTY BUILD_WITH_INSTALL_RPATH FALSE)
      target_link_libraries("${target}" PRIVATE ${CMAKE_DL_LIBS})
    endif()
  endforeach()
  get_property(
    directories
    DIRECTORY "${directory}"
    PROPERTY SUBDIRECTORIES)
  foreach(child IN LISTS directories)
    mqt_study_test_runtime("${child}")
  endforeach()
endfunction()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_language(DEFER CALL mqt_study_test_runtime "${CMAKE_SOURCE_DIR}")
endif()
