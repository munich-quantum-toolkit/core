# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device-gen)

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})
  # ------------------------------------------------------------------------------
  # Generate C++ source files from .proto
  # ------------------------------------------------------------------------------

  # Proto files
  set(PROTO_FILE ${PROJECT_SOURCE_DIR}/proto/na/device.proto)

  # Set paths
  set(PROTO_SRC ${CMAKE_CURRENT_BINARY_DIR}/device.pb.cc)
  set(PROTO_HDR ${CMAKE_CURRENT_BINARY_DIR}/device.pb.h)

  # Generate .pb.cc and .pb.h
  add_custom_command(
    OUTPUT ${PROTO_SRC} ${PROTO_HDR}
    COMMAND protobuf::protoc ARGS --proto_path=${PROJECT_SOURCE_DIR}/proto/na
            --cpp_out=${CMAKE_CURRENT_BINARY_DIR} ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating C++ source from ${PROTO_FILE}")
  add_custom_target(generate_proto_files DEPENDS ${PROTO_SRC} ${PROTO_HDR})

  # Add executable for device generation
  add_executable(${TARGET_NAME})
  add_dependencies(${TARGET_NAME} generate_proto_files)

  # add sources to target
  target_sources(${TARGET_NAME} PRIVATE Generator.cpp App.cpp ${PROTO_SRC})

  # add headers using file sets
  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${MQT_CORE_INCLUDE_BUILD_DIR}
           ${CMAKE_BINARY_DIR}/src
           FILES
           ${MQT_CORE_INCLUDE_BUILD_DIR}/na/device/Generator.hpp
           ${PROTO_HDR})

  # Link Protobuf, QDMI, spdlog, and project warnings
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC protobuf::libprotobuf spdlog::spdlog
    PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  # set versioning information
  set_target_properties(${TARGET_NAME} PROPERTIES VERSION ${PROJECT_VERSION} EXPORT_NAME
                                                                             CoreNaDeviceGen)
  # place in the bin directory
  set_target_properties(${TARGET_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  "${CMAKE_BINARY_DIR}/bin")

  # Make version available
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${TARGET_NAME}
      PARENT_SCOPE)

  # Create an alias for the target
  add_executable(MQT::CoreNADeviceGen ALIAS ${TARGET_NAME})
endif()

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device)

# Set prefix for QDMI
set(QDMI_PREFIX "MQT_NA")

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Set paths
  set(JSON_FILE ${PROJECT_SOURCE_DIR}/json/na/device.json)
  set(DEVICE_HDR ${CMAKE_CURRENT_BINARY_DIR}/include/Device.hpp)

  # Generate definitions for device
  add_custom_command(
    OUTPUT ${DEVICE_HDR}
    COMMAND MQT::CoreNADeviceGen ARGS --output ${DEVICE_HDR} ${JSON_FILE}
    DEPENDS ${JSON_FILE}
    COMMENT "Generating C++ header from ${JSON_FILE}")
  add_custom_target(generate_na_device_header DEPENDS ${DEVICE_HDR})

  # Add library
  add_mqt_core_library(${TARGET_NAME} ALIAS_NAME NaQDMI)
  add_dependencies(${TARGET_NAME} generate_na_device_header)

  # Generate prefixed QDMI headers
  generate_prefixed_qdmi_headers(${QDMI_PREFIX})
  file(GLOB_RECURSE QDMI_HDRS ${CMAKE_CURRENT_BINARY_DIR}/include/mqt_na_qdmi/**.hpp)

  # add sources to target
  target_sources(${TARGET_NAME} PRIVATE Device.cpp)

  # add headers using file sets
  target_sources(
    ${TARGET_NAME}
    PUBLIC FILE_SET
           HEADERS
           BASE_DIRS
           ${CMAKE_CURRENT_BINARY_DIR}/include
           FILES
           ${DEVICE_HDR}
           ${QDMI_HDRS})

  # add link libraries
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi spdlog::spdlog
    PRIVATE MQT::ProjectOptions MQT::ProjectWarnings)

  # set c++ standard
  target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)

  # set versioning information
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreNaQDMI)

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${TARGET_NAME}
      PARENT_SCOPE)

  # Make QDMI and MQT Core Version available
  target_compile_definitions(${TARGET_NAME} PRIVATE QDMI_VERSION="${QDMI_VERSION}")
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")

  # Generate additional alias for the target
  add_library(qdmi::mqt_na_device ALIAS ${TARGET_NAME})
endif()
