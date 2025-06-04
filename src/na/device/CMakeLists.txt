include(FindProtobuf)

# Set target name
set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-na-device)

# If the target is not already defined
if(NOT TARGET ${TARGET_NAME})

  # todo: This seems wrong to me, does MQT Core not already set the C++ standard?
  # Set the C++ standard
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  # Proto files
  set(PROTO_FILES ${PROJECT_SOURCE_DIR}/proto/na/device.proto)

  # Generate C++ source files from .proto
  protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

  # Add executable
  add_library(${TARGET_NAME} device.cpp ${PROTO_SRCS} ${PROTO_HDRS})

  # Add include directories
  target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_include_directories(${TARGET_NAME} PRIVATE ${Protobuf_INCLUDE_DIRS})

  # Link Protobuf
  target_link_libraries(${TARGET_NAME} PRIVATE ${Protobuf_LIBRARIES})

endif()