# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(common)
add_subdirectory(devices)
add_subdirectory(driver)

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi)

if(NOT TARGET ${TARGET_NAME})
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMI)
  set_property(TARGET ${TARGET_NAME} PROPERTY BUILD_WITH_INSTALL_RPATH FALSE)

  target_sources(${TARGET_NAME} PRIVATE Client.cpp Slurm.cpp)

  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/Client.hpp
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/Slurm.hpp)

  target_link_libraries(${TARGET_NAME} PUBLIC qdmi::qdmi MQT::CoreQDMICommon MQT::CoreQDMIDriver)

  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

if(UNIX)
  add_executable(mqt-core-qdmi-check Check.cpp)
  set_property(TARGET mqt-core-qdmi-check PROPERTY BUILD_WITH_INSTALL_RPATH FALSE)
  target_link_libraries(mqt-core-qdmi-check PRIVATE MQT::CoreQDMI MQT::ProjectWarnings
                                                    MQT::ProjectOptions)
  mqt_get_qdmi_device_targets(checker_devices)
  if(checker_devices AND NOT BUILD_MQT_CORE_SHARED_LIBS)
    mqt_copy_qdmi_runtime(mqt-core-qdmi-check ${checker_devices})
  endif()
  if(MQT_CORE_INSTALL)
    if(BUILD_MQT_CORE_SHARED_LIBS)
      set(checker_install_dir "${CMAKE_INSTALL_BINDIR}")
    else()
      # A static driver discovers catalogues beside its executable.
      set(checker_install_dir "${CMAKE_INSTALL_LIBDIR}")
      file(RELATIVE_PATH checker_runtime_dir "${CMAKE_INSTALL_FULL_BINDIR}"
           "${CMAKE_INSTALL_FULL_LIBDIR}")
      set(checker_launcher "${CMAKE_CURRENT_BINARY_DIR}/mqt-core-qdmi-check-launcher")
      file(
        GENERATE
        OUTPUT "${checker_launcher}"
        CONTENT "#!/bin/sh\nexec \"\${0%/*}/${checker_runtime_dir}/mqt-core-qdmi-check\" \"$@\"\n")
      install(
        PROGRAMS "${checker_launcher}"
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        RENAME mqt-core-qdmi-check
        COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
    endif()
    install(TARGETS mqt-core-qdmi-check RUNTIME DESTINATION "${checker_install_dir}"
                                                COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
  endif()
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
