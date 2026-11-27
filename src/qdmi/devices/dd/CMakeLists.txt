# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-ddsim-device)

set(QDMI_PREFIX "MQT_DDSIM")

if(NOT TARGET ${TARGET_NAME})

  add_mqt_core_library(${TARGET_NAME} FORCE_SHARED HIDDEN_VISIBILITY ALIAS_NAME QDMI_DDSIM_Device)

  generate_prefixed_qdmi_headers(${QDMI_PREFIX})
  file(GLOB_RECURSE QDMI_HDRS ${CMAKE_CURRENT_BINARY_DIR}/include/mqt_ddsim_qdmi/**.h)

  if(NOT LLVM_ENABLE_THREADS)
    message(FATAL_ERROR "The DDSIM QDMI device requires a threaded LLVM build")
  endif()
  target_sources(
    ${TARGET_NAME}
    PRIVATE Device.cpp Worker.cpp WorkerProtocol.cpp WorkerTransport.cpp)

  target_sources(
    ${TARGET_NAME}
    PUBLIC
    FILE_SET HEADERS
      BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/include
      FILES ${QDMI_HDRS})

  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreDD MQT::CoreQDMICommon LLVMSupport)
  mqt_llvm_target_disable_rtti(${TARGET_NAME})
  # Export only the QDMI interface. A plugin-enabled host could otherwise interpose LLVM's
  # process-wide registries from static dependencies when it loads this device.
  configure_qdmi_exports(TARGET ${TARGET_NAME} INTERFACE device PREFIX ${QDMI_PREFIX})
  if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)
  endif()
  file(RELATIVE_PATH simulator_path "/${CMAKE_INSTALL_LIBDIR}"
       "/${CMAKE_INSTALL_BINDIR}/mqt-core${CMAKE_EXECUTABLE_SUFFIX}")
  target_compile_definitions(
    ${TARGET_NAME}
    PRIVATE
      MQT_DDSIM_EXECUTABLE="mqt-core${CMAKE_EXECUTABLE_SUFFIX}"
      MQT_DDSIM_EXECUTABLE_RELATIVE="${simulator_path}")
  mqt_configure_qdmi_device(${TARGET_NAME} ID mqt.ddsim.default PREFIX ${QDMI_PREFIX})
  set_property(TARGET ${TARGET_NAME} PROPERTY QDMI_RUNTIME_TARGETS MQT::mqt-core)
  set_property(TARGET ${TARGET_NAME} APPEND PROPERTY EXPORT_PROPERTIES QDMI_RUNTIME_TARGETS)
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS ${MQT_CORE_TARGETS} PARENT_SCOPE)
