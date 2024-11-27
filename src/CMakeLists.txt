add_subdirectory(qir)

if(NOT TARGET MQT::MLIR)
  # collect headers and source files
  file(GLOB MQT_MLIR_HEADERS ${MQT_MLIR_INCLUDE_BUILD_DIR}/*.hpp)
  file(GLOB MQT_MLIR_SOURCES **.cpp)

  # add MLIR Package library
  add_library(${PROJECT_NAME} ${MQT_MLIR_HEADERS} ${MQT_MLIR_SOURCES})

  # set include directories
  target_include_directories(${PROJECT_NAME}
                             PUBLIC $<BUILD_INTERFACE:${MQT_MLIR_INCLUDE_BUILD_DIR}>)

  # link to the MQT::Core libraries
  target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC MQT::CoreIR
    PRIVATE MQT::ProjectWarnings MQT::ProjectOptions)

  # add MQT alias
  add_library(MQT::MLIR ALIAS ${PROJECT_NAME})
endif()
