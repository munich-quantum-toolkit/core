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
           ${MQT_CORE_INCLUDE_BUILD_DIR}/na/device_generator/Generator.hpp
           ${PROTO_HDR})

  # Link Protobuf, QDMI, spdlog, and project warnings
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC protobuf::libprotobuf spdlog::spdlog
    PRIVATE qdmi::project_warnings MQT::ProjectOptions MQT::ProjectWarnings)

  # set versioning information
  set_target_properties(
    ${TARGET_NAME}
    PROPERTIES VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
               EXPORT_NAME CoreNaDeviceGen)

  # Make version available
  target_compile_definitions(${TARGET_NAME} PRIVATE MQT_CORE_VERSION="${MQT_CORE_VERSION}")

  # add to list of MQT core targets
  set(MQT_CORE_TARGETS
      ${MQT_CORE_TARGETS} ${TARGET_NAME}
      PARENT_SCOPE)

  # Create an alias for the target
  add_executable(MQT::CoreNADeviceGen ALIAS ${TARGET_NAME})
endif()
