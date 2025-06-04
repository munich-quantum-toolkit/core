# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

include(FindProtobuf)

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device)

# Set prefix for QDMI
set(QDMI_PREFIX "MQT_NA")

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # todo: This seems wrong to me, does MQT Core not already set the C++ standard? Set the C++
  # standard
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  # Proto files
  set(PROTO_FILES ${PROJECT_SOURCE_DIR}/proto/na/device.proto)

  # Generate C++ source files from .proto
  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

  # Add executable
  add_library(${TARGET_NAME} device.cpp ${PROTO_SRCS} ${PROTO_HDRS})

  # Generate prefixed QDMI headers
  generate_prefixed_qdmi_headers(${QDMI_PREFIX})

  # Add include directories
  target_include_directories(
    ${TARGET_NAME}
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${Protobuf_INCLUDE_DIRS})

  # Link Protobuf, QDMI, spdlog, and project warnings
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi ${Protobuf_LIBRARIES} spdlog::spdlog
    PRIVATE qdmi::project_warnings MQT::ProjectOptions MQT::ProjectWarnings)

  # Generate alias for the target
  add_library(MQT::CoreNaQDMI ALIAS ${TARGET_NAME})
  add_library(qdmi::mqt_na_device ALIAS ${TARGET_NAME})
endif()
