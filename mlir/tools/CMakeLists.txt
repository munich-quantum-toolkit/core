# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(mqt-cc NativeCompilerMain.cpp mqt-cc/mqt-cc.cpp ../bench/MQTCoreBench.cpp)
add_executable(MQT::mqt-cc ALIAS mqt-cc)
target_include_directories(mqt-cc PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
mqt_mlir_target_use_project_options(mqt-cc)
target_link_libraries(
  mqt-cc
  PRIVATE
    MQT::ProjectWarnings
    MQT::CoreBenchGenerate
    MQTCompilerQDMIAdapter
    MQTCompilerPipeline
    MLIRParser
    MLIRSupport
    MLIRQCOpenQASMTranslation
    MLIRJeffTranslation
    MLIRJeffToQCO
    MLIRBytecodeWriter
    MLIRQIRTranslation)
if(TARGET ${MQT_CORE_TARGET_NAME}-qdmi-ddsim-device)
  target_sources(
    mqt-cc
    PRIVATE ../../src/qdmi/devices/dd/WorkerMain.cpp ../../src/qdmi/devices/dd/WorkerProtocol.cpp)
  target_compile_definitions(mqt-cc PRIVATE MQT_CORE_HAS_SIMULATOR)
  target_link_libraries(
    mqt-cc
    PRIVATE
      MQT::CoreDD MQT::CoreQDMICommon MQT::CoreQIRJIT MQT::CoreQIRRuntime MLIRQCODDFunctionality
      MQTCompilerPrograms)
endif()
mlir_check_all_link_libraries(mqt-cc)
set_target_properties(
  mqt-cc
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
add_dependencies(mqt-cc ${MQT_CORE_TARGET_NAME}-qdmi-driver)
if(MQT_CORE_INSTALL)
  install(
    TARGETS mqt-cc
    EXPORT mqt-core-runtime-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
