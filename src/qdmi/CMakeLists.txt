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

  target_sources(${TARGET_NAME} PRIVATE QDMI.cpp Slurm.cpp)

  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/QDMI.hpp
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/Slurm.hpp)

  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreQDMICommon
    PRIVATE ${CMAKE_DL_LIBS})

  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE "MQT_CORE_QDMI_DEFAULT_DRIVER_FILENAME=\"$<TARGET_FILE_NAME:MQT::CoreQDMIDriver>\"")
  add_dependencies(${TARGET_NAME} ${MQT_CORE_TARGET_NAME}-qdmi-driver)

  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

if(UNIX OR WIN32)
  add_executable(mqt-core-qdmi-check Check.cpp)
  target_link_libraries(mqt-core-qdmi-check PRIVATE MQT::CoreQDMI MQT::ProjectWarnings
                                                    MQT::ProjectOptions)
  if(WIN32)
    target_compile_definitions(mqt-core-qdmi-check PRIVATE NOMINMAX _WIN32_WINNT=0x0A00)
    if(MINGW)
      target_link_options(mqt-core-qdmi-check PRIVATE -municode)
    endif()
  endif()
  mqt_copy_qdmi_runtime(mqt-core-qdmi-check)
  if(MQT_CORE_INSTALL)
    install(TARGETS mqt-core-qdmi-check RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
                                                COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
  endif()
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
