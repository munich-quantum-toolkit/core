# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME mqt-core-qdmi-object-model-test)

if(TARGET MQT::CoreQDMI)
  package_add_test(${TARGET_NAME} MQT::CoreQDMI test_device.cpp test_device_api.cpp)
  target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src/qdmi)
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMICommon qdmi::qdmi)
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE DDSIM_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMI_DDSIM_Device>"
            NA_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMINaDevice>"
            SC_DEVICE_LIBRARY="$<TARGET_FILE:MQT::CoreQDMIScDevice>")
  add_dependencies(${TARGET_NAME} MQT::CoreQDMI_DDSIM_Device MQT::CoreQDMINaDevice
                   MQT::CoreQDMIScDevice)

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
