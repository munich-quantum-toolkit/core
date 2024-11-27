set(MQT_QIR_BACKEND_TARGET_NAME ${PROJECT_NAME}-qir-backend)

if(NOT TARGET ${MQT_QIR_BACKEND_TARGET_NAME})
  file(GLOB_RECURSE QIR_BACKEND_HEADERS ${PROJECT_SOURCE_DIR}/include/qir/*.h
       ${PROJECT_SOURCE_DIR}/include/qir/*.hpp)
  file(GLOB_RECURSE QIR_BACKEND_SOURCES *.c *.cpp)

  add_library(${MQT_QIR_BACKEND_TARGET_NAME} ${QIR_BACKEND_HEADERS} ${QIR_BACKEND_SOURCES})

  # set include directories
  target_include_directories(${MQT_QIR_BACKEND_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

  # link to the MQT::Core libraries
  target_link_libraries(
    ${MQT_QIR_BACKEND_TARGET_NAME}
    PUBLIC MQT::CoreIR MQT::CoreDD
    PRIVATE MQT::ProjectWarnings MQT::ProjectOptions)

  add_library(MQT::QIRBackend ALIAS ${MQT_QIR_BACKEND_TARGET_NAME})
endif()
