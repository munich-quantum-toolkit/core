# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(TARGET_NAME ${MQT_CORE_TARGET_NAME}-qdmi-driver)

if(NOT TARGET ${TARGET_NAME})

  # Check whether MQT::CoreNaQDMI is a shared library or not. If it is a shared library, we will use
  # it directly. Otherwise, we will duplicate the target as a shared library.
  get_target_property(TARGET_TYPE MQT::CoreNaQDMI TYPE)
  if(TARGET_TYPE STREQUAL "SHARED_LIBRARY")
    set(DEVICE_TARGET MQT::CoreNaQDMI)
  else()
    # Duplicate target as a shared library
    set(DEVICE_TARGET ${MQT_CORE_TARGET_NAME}-na-qdmi-shared)
    add_library(${DEVICE_TARGET} SHARED)
    target_sources(${DEVICE_TARGET} PRIVATE $<TARGET_OBJECTS:MQT::CoreNaQDMI>)
    get_target_property(INCLUDE_DIRS MQT::CoreNaQDMI INCLUDE_DIRECTORIES)
    target_include_directories(${DEVICE_TARGET} PRIVATE ${INCLUDE_DIRS})
    get_target_property(COMPILE_DEFINITIONS MQT::CoreNaQDMI COMPILE_DEFINITIONS)
    target_compile_definitions(${DEVICE_TARGET} PRIVATE ${COMPILE_DEFINITIONS})
    get_target_property(COMPILE_OPTIONS MQT::CoreNaQDMI COMPILE_OPTIONS)
    target_compile_options(${DEVICE_TARGET} PRIVATE ${COMPILE_OPTIONS})
    get_target_property(LINK_LIBRARIES MQT::CoreNaQDMI INTERFACE_LINK_LIBRARIES)
    target_link_libraries(${DEVICE_TARGET} PRIVATE ${LINK_LIBRARIES})
  endif()

  add_mqt_core_library(${TARGET_NAME})
  add_dependencies(${TARGET_NAME} ${DEVICE_TARGET})
  target_sources(${TARGET_NAME} PRIVATE driver.cpp)
  target_sources(${TARGET_NAME} PUBLIC FILE_SET HEADERS BASE_DIRS ${MQT_CORE_INCLUDE_BUILD_DIR}
                                       FILES ${MQT_CORE_INCLUDE_BUILD_DIR}/qdmi/driver.hpp)
  target_link_libraries(
    ${TARGET_NAME}
    PUBLIC qdmi::qdmi MQT::CoreNaQDMI
    PRIVATE qdmi::project_warnings MQT::ProjectOptions MQT::ProjectWarnings)
  set_target_properties(${TARGET_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_compile_definitions(${TARGET_NAME} PRIVATE KNOWN_LIBRARIES="libmqt-core-na-device.dylib")
  target_compile_definitions(${TARGET_NAME} PRIVATE KNOWN_LIBRARY_PREFIXES="MQT_NA")

endif()
