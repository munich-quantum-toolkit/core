# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-bindings)
nanobind_add_module(
  ${TARGET_NAME}
  FREE_THREADED
  NOMINSIZE
  NB_SUPPRESS_WARNINGS
  BACKEND_MODULE
  nanobind_backend
  register_core.cpp)
set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME _core)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

# Hide archive symbols without hiding C++ exception type information on Mach-O.
if(UNIX AND NOT APPLE)
  target_link_options(${TARGET_NAME} PRIVATE "LINKER:--exclude-libs,ALL")
endif()

if(MQT_CORE_INSTALL)
  install(TARGETS ${TARGET_NAME} DESTINATION . COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()

add_subdirectory(bench)
add_subdirectory(dd)
add_subdirectory(qdmi)
add_subdirectory(mlir)
