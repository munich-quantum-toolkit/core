# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-driver-test)

if(TARGET MQT::CoreQDMIDriver)
  package_add_test(${TARGET_NAME} MQT::CoreQDMIDriver test_driver.cpp)
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreFoMaC)
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE
      "DYN_DEV_LIBS=std::array{ std::pair{\"$<TARGET_FILE_NAME:MQT::CoreQDMINaDevice>\", \"MQT_NA\"}, std::pair{\"$<TARGET_FILE_NAME:MQT::CoreQDMIScDevice>\", \"MQT_SC\"}, std::pair{\"$<TARGET_FILE_NAME:MQT::CoreQDMI_DDSIM_Device>\", \"MQT_DDSIM\"} }"
  )

  if(WIN32)
    # On Windows, we need to copy the device DLLs to the test directory
    add_custom_command(
      TARGET ${TARGET_NAME}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:MQT::CoreQDMIScDevice>
              $<TARGET_FILE_DIR:${TARGET_NAME}>)
    add_custom_command(
      TARGET ${TARGET_NAME}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:MQT::CoreQDMINaDevice>
              $<TARGET_FILE_DIR:${TARGET_NAME}>)
    add_custom_command(
      TARGET ${TARGET_NAME}
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>
              $<TARGET_FILE_DIR:${TARGET_NAME}>)
  endif()
endif()
