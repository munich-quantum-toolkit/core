# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-sc-device-gen)

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Add library for device generation
  #
  # Note: We use a static library here to avoid issues with RPATH and finding the executable during
  # the build process in Python builds
  add_library(${TARGET_NAME} STATIC)
  add_library(MQT::CoreQDMIScDeviceGen ALIAS ${TARGET_NAME})

  # add sources to target
  target_sources(${TARGET_NAME} PRIVATE Generator.cpp)

  # add headers using file sets
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/sc/Generator.hpp)

  # Link nlohmann_json, spdlog
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC nlohmann_json::nlohmann_json
    PRIVATE spdlog::spdlog MQT::ProjectOptions MQT::ProjectWarnings)

  # set versioning information
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreScDeviceGen)

  # set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS ${MQT_CORE_TARGETS} ${TARGET_NAME})

  # Make version available
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")
endif()

# Set target name
set(TARGET_NAME mqt-core-qdmi-sc-device-generator)

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})
  # Add executable for device generation
  add_executable(${TARGET_NAME})

  # add sources to target
  target_sources(${TARGET_NAME} PRIVATE App.cpp)

  # Link Generator library
  target_link_libraries(${TARGET_NAME} PRIVATE MQT::CoreQDMIScDeviceGen spdlog::spdlog)

  # set versioning information
  set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION} EXPORT_NAME
                                                                             CoreScDeviceGen)
  # place in the bin directory
  set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${CMAKE_BINARY_DIR}/bin")

  # set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)

  # Make version available
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")

  # Create an alias for the target
  add_executable(MQT::CoreQDMIScDeviceGenerator ALIAS ${TARGET_NAME})
endif()

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-sc-device)

# Set prefix for QDMI
set(QDMI_PREFIX "MQT_SC")

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Set paths
  set(JSON_FILE ${PROJECT_SOURCE_DIR}/json/sc/device.json)
  set(DEVICE_HDR ${CMAKE_CURRENT_BINARY_DIR}/include/qdmi/sc/DeviceMemberInitializers.hpp)

  # Create include directory
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/qdmi/sc)

  # Generate definitions for device
  add_custom_command(
    OUTPUT ${DEVICE_HDR}
    COMMAND MQT::CoreQDMIScDeviceGenerator ARGS generate --output ${DEVICE_HDR} ${JSON_FILE}
    DEPENDS ${JSON_FILE} MQT::CoreQDMIScDeviceGenerator
    COMMENT "Generating C++ header from ${JSON_FILE}")
  add_custom_target(generate_qdmi_sc_device_header DEPENDS ${DEVICE_HDR})

  # Add library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME QDMIScDevice)
  add_dependencies(${TARGET_NAME} generate_qdmi_sc_device_header)

  # Generate prefixed QDMI headers
  generate_prefixed_qdmi_headers(${QDMI_PREFIX})
  file(GLOB_RECURSE QDMI_HDRS ${CMAKE_CURRENT_BINARY_DIR}/include/mqt_sc_qdmi/**.h)

  # add sources to target
  target_sources(${TARGET_NAME} PRIVATE Device.cpp)

  # add headers using file sets
  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           ${CMAKE_CURRENT_BINARY_DIR}/include
           FILES
           ${DEVICE_HDR}
           ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/sc/Device.hpp
           ${QDMI_HDRS})

  # add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi
    PRIVATE MQT::CoreQDMICommon spdlog::spdlog)

  # Always compile with position independent code such that the library can be used in shared
  # libraries
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # add to list of MQT core targets
  list(APPEND MQT_CORE_TARGETS ${TARGET_NAME})

  # Make QDMI and MQT Core Version available
  target_compile_definitions(${TARGET_NAME} PRIVATE QDMI_VERSION="${QDMI_VERSION}")

  # Generate additional alias for the target required for generate_device_defs_executable function
  # in the tests
  add_library(qdmi::mqt_sc_device ALIAS ${TARGET_NAME})

  # Do not build dynamic SC device on Windows because it cannot be used anyways in the current setup
  if(NOT WIN32)
    set(DYN_TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-sc-device-dyn)
    if(NOT TARGET ${DYN_TARGET_NAME})
      # Set prefix for QDMI
      set(QDMI_PREFIX "MQT_SC_DYN")
      # Generate prefixed QDMI headers
      generate_prefixed_qdmi_headers(${QDMI_PREFIX})
      file(GLOB_RECURSE QDMI_HDRS ${CMAKE_CURRENT_BINARY_DIR}/include/mqt_sc_dyn_qdmi/**.hpp)
      # Add dynamic library target
      add_library(${DYN_TARGET_NAME} SHARED)
      add_dependencies(${DYN_TARGET_NAME} ${TARGET_NAME})
      # add sources to target
      target_sources(${DYN_TARGET_NAME} PRIVATE DynDevice.cpp)
      # add headers using file sets
      target_sources(
        ${DYN_TARGET_NAME}
        PUBLIC FILE_SET
               HEADERS
               BASE_DIRS
               ${MQT_CORE_INCLUDE_BUILD_DIR}
               ${CMAKE_CURRENT_BINARY_DIR}/include
               FILES
               ${DEVICE_HDR}
               ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/sc/Device.hpp
               ${QDMI_HDRS})
      # add link libraries
      target_link_libraries(
        ${DYN_TARGET_NAME}
        PUBLIC qdmi::qdmi
        PRIVATE ${TARGET_NAME} MQT::CoreQDMICommon MQT::ProjectOptions MQT::ProjectWarnings
                spdlog::spdlog)
      # set c++ standard
      target_compile_features(${DYN_TARGET_NAME} PRIVATE cxx_std_20)
      # set versioning information
      set_target_properties(
        ${DYN_TARGET_NAME}
        PROPERTIES VERSION ${PROJECT_VERSION}
                   SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
                   EXPORT_NAME CoreQDMIScDeviceDyn)
      add_library(MQT::CoreQDMIScDeviceDyn ALIAS ${DYN_TARGET_NAME})
      list(APPEND MQT_CORE_TARGETS ${DYN_TARGET_NAME})
    endif()
  endif()
endif()
set(MQT_CORE_TARGETS
    ${MQT_CORE_TARGETS}
    PARENT_SCOPE)
