# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_executable(
  mqt-core
  NativeMain.cpp
  simulate/Simulate.cpp
  compile/Compile.cpp
  ../bench/MQTCoreBench.cpp
  ../../src/qdmi/devices/dd/WorkerMain.cpp
  ../../src/qdmi/devices/dd/WorkerProtocol.cpp)
add_executable(MQT::mqt-core ALIAS mqt-core)
target_include_directories(mqt-core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
mqt_mlir_target_use_project_options(mqt-core)
target_link_libraries(
  mqt-core
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
    MLIRQIRTranslation
    MQT::CoreDD
    MQT::CoreQDMICommon
    MQT::CoreQIRJIT
    MQT::CoreQIRRuntime
    MLIRQCODDFunctionality
    MQTCompilerPrograms)
file(RELATIVE_PATH include_relative "${CMAKE_INSTALL_FULL_BINDIR}"
     "${CMAKE_INSTALL_FULL_INCLUDEDIR}/mqt-core")
file(RELATIVE_PATH cmake_relative "${CMAKE_INSTALL_FULL_BINDIR}"
     "${CMAKE_INSTALL_FULL_DATADIR}/cmake")
target_compile_definitions(
  mqt-core
  PRIVATE
    MQT_CORE_VERSION_STRING="${MQT_CORE_VERSION_STRING}"
    MQT_CORE_INCLUDE_RELATIVE="${include_relative}" MQT_CORE_CMAKE_RELATIVE="${cmake_relative}")
mlir_check_all_link_libraries(mqt-core)
set_target_properties(
  mqt-core
  PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
add_dependencies(mqt-core ${MQT_CORE_TARGET_NAME}-qdmi-driver)
if(MQT_CORE_INSTALL)
  install(
    TARGETS mqt-core
    EXPORT mqt-core-runtime-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${MQT_CORE_TARGET_NAME}_Runtime)
endif()
