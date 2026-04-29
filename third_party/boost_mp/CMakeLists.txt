# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_CORE_BOOST_MP_VERSION
    1.89.0
    CACHE STRING "Vendored boost multiprecision version")

set(MQT_CORE_BOOST_MP_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(MQT_CORE_BOOST_MP_INCLUDE_DIR "${MQT_CORE_BOOST_MP_ROOT_DIR}/include")
set(MQT_CORE_BOOST_MP_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}/mqt-core/third_party/boost_mp/include")
set(MQT_CORE_BOOST_MP_TARGET "mqt-core-vendor-boost-mp")

if(NOT EXISTS "${MQT_CORE_BOOST_MP_INCLUDE_DIR}/boost/multiprecision/cpp_int.hpp")
  message(FATAL_ERROR "Vendored boost multiprecision header not found at "
                      "${MQT_CORE_BOOST_MP_INCLUDE_DIR}/boost/multiprecision/cpp_int.hpp")
endif()

if(NOT TARGET ${MQT_CORE_BOOST_MP_TARGET})
  add_library(${MQT_CORE_BOOST_MP_TARGET} INTERFACE)
  add_library(MQT::Multiprecision ALIAS ${MQT_CORE_BOOST_MP_TARGET})
  add_library(MQT::multiprecision ALIAS ${MQT_CORE_BOOST_MP_TARGET})
  set_target_properties(${MQT_CORE_BOOST_MP_TARGET} PROPERTIES EXPORT_NAME Multiprecision)

  target_include_directories(
    ${MQT_CORE_BOOST_MP_TARGET} SYSTEM
    INTERFACE $<BUILD_INTERFACE:${MQT_CORE_BOOST_MP_INCLUDE_DIR}>
              $<INSTALL_INTERFACE:${MQT_CORE_BOOST_MP_INCLUDE_INSTALL_DIR}>)
endif()

if(MQT_CORE_INSTALL)
  install(
    TARGETS ${MQT_CORE_BOOST_MP_TARGET}
    EXPORT ${MQT_CORE_THIRD_PARTY_TARGETS_EXPORT_NAME}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  install(
    DIRECTORY ${MQT_CORE_BOOST_MP_INCLUDE_DIR}/
    DESTINATION ${MQT_CORE_BOOST_MP_INCLUDE_INSTALL_DIR}
    COMPONENT ${MQT_CORE_TARGET_NAME}_Development)

  if(EXISTS "${MQT_CORE_BOOST_MP_ROOT_DIR}/LICENSE")
    install(
      FILES ${MQT_CORE_BOOST_MP_ROOT_DIR}/LICENSE
      DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/mqt-core/boost_mp
      COMPONENT ${MQT_CORE_TARGET_NAME}_Development)
  endif()
endif()
