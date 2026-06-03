# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 MQSC GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-fomac-test)

if(TARGET MQT::CoreFoMaC)
  package_add_test(${TARGET_NAME} MQT::CoreFoMaC test_fomac.cpp)

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
