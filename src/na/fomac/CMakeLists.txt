# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-fomac-na)

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Add library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME FoMaCNa)

  # add sources to target
  target_sources(${TARGET_NAME} PRIVATE Device.cpp)

  # add headers using file sets
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/na/fomac/Device.hpp)

  # Link nlohmann_json, spdlog
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC MQT::CoreQDMIFoMaC
    PRIVATE spdlog::spdlog)

  # set versioning information
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreFoMaCNa)

  # set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
endif()
