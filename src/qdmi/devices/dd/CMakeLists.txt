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
  target_sources(${TARGET_NAME} PRIVATE Device.cpp Worker.cpp WorkerProtocol.cpp)

  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           ${CMAKE_CURRENT_BINARY_DIR}/include
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/devices/dd/Device.hpp
           ${QDMI_HDRS})

  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreDD MQT::CoreQDMICommon LLVMSupport)
  mqt_llvm_target_disable_rtti(${TARGET_NAME})
  # Keep symbols from static dependencies local. A plugin-enabled host can otherwise interpose
  # LLVM's process-wide registries when it loads this device.
  if(APPLE)
    target_link_options(${TARGET_NAME} PRIVATE "LINKER:-exported_symbol,_${QDMI_PREFIX}_QDMI_*")
  elseif(UNIX)
    target_link_options(${TARGET_NAME} PRIVATE "LINKER:--exclude-libs,ALL")
  endif()
  set(WORKER_TARGET ${MQT_CORE_TARGET_NAME}-ddsim-worker)
  add_executable(${WORKER_TARGET} WorkerMain.cpp WorkerProtocol.cpp)
  target_compile_features(${WORKER_TARGET} PRIVATE cxx_std_20)
  target_link_libraries(
    ${WORKER_TARGET}
    PRIVATE MQT::CoreDD
            MQT::CoreQDMICommon
            MLIRQCODDFunctionality
            MQTCompilerPrograms
            MQT::CoreQIRJIT
            MQT::CoreQIRRuntime
            MQT::ProjectWarnings
            MQT::ProjectOptions)
  mqt_llvm_target_disable_rtti(${WORKER_TARGET})
  if(WIN32)
    target_compile_definitions(${TARGET_NAME} PRIVATE NOMINMAX)
    set(WORKER_DIRECTORY ${CMAKE_INSTALL_BINDIR})
  else()
    set(WORKER_DIRECTORY ${CMAKE_INSTALL_LIBDIR})
  endif()
  set_target_properties(${WORKER_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    "${CMAKE_BINARY_DIR}/${WORKER_DIRECTORY}")
  target_compile_definitions(
    ${TARGET_NAME} PRIVATE MQT_DDSIM_WORKER_NAME="${WORKER_TARGET}${CMAKE_EXECUTABLE_SUFFIX}")
  add_dependencies(${TARGET_NAME} ${WORKER_TARGET})
  install(TARGETS ${WORKER_TARGET} RUNTIME DESTINATION ${WORKER_DIRECTORY}
                                           COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
  mqt_configure_qdmi_device(${TARGET_NAME} ID mqt.ddsim.default PREFIX ${QDMI_PREFIX})
  set_property(
    TARGET ${TARGET_NAME}
    APPEND
    PROPERTY QDMI_RUNTIME_FILES "${WORKER_TARGET}${CMAKE_EXECUTABLE_SUFFIX}")
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})
endif()

set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
