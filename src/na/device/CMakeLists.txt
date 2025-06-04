# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device)

# Set prefix for QDMI
set(QDMI_PREFIX "MQT_NA")

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # Proto files
  set(PROTO_FILE ${PROJECT_SOURCE_DIR}/proto/na/device.proto)

  # ------------------------------------------------------------------------------
  # Generate C++ source files from .proto
  # ------------------------------------------------------------------------------

  # protobuf_generate_cpp(PROTO_SRC PROTO_HDR ${PROTO_FILES})

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

  # Add executable
  add_library(${TARGET_NAME} device.cpp ${PROTO_SRC} ${PROTO_HDR})
  add_dependencies(${TARGET_NAME} generate_proto_files)

  # Generate prefixed QDMI headers
  generate_prefixed_qdmi_headers(${QDMI_PREFIX})

  # Add include directories
  target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include
                                                   ${CMAKE_CURRENT_BINARY_DIR})

  # Link Protobuf, QDMI, spdlog, and project warnings
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi protobuf::libprotobuf spdlog::spdlog
    PRIVATE qdmi::project_warnings MQT::ProjectOptions MQT::ProjectWarnings)

  # Generate alias for the target
  add_library(MQT::CoreNaQDMI ALIAS ${TARGET_NAME})
  add_library(qdmi::mqt_na_device ALIAS ${TARGET_NAME})
endif()
