# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of the optional MQT Core SPANK component.
# See LICENSE.md for the GNU General Public License, version 3 or later.

cmake_minimum_required(VERSION 3.28)
project(mqt-core-qdmi-spank LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "The MQT Core SPANK component requires Linux and Slurm 25.11 or newer")
endif()

include(GNUInstallDirs)
find_path(SLURM_SPANK_INCLUDE_DIR slurm/spank.h REQUIRED)

add_library(mqt-core-qdmi-spank MODULE spank.cpp)
target_compile_features(mqt-core-qdmi-spank PRIVATE cxx_std_20)
target_include_directories(mqt-core-qdmi-spank PRIVATE ${SLURM_SPANK_INCLUDE_DIR})
set_target_properties(mqt-core-qdmi-spank PROPERTIES PREFIX "")
target_link_options(mqt-core-qdmi-spank PRIVATE
                    "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/mqt-core-qdmi-spank.map")
set_property(
  TARGET mqt-core-qdmi-spank
  APPEND
  PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mqt-core-qdmi-spank.map")

set(MQT_CORE_SPANK_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/slurm"
    CACHE PATH "Install directory for the Slurm SPANK module")
install(TARGETS mqt-core-qdmi-spank LIBRARY DESTINATION ${MQT_CORE_SPANK_INSTALL_DIR}
                                            COMPONENT mqt-core-qdmi-spank)
install(
  FILES LICENSE.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/mqt-core-qdmi-spank
  COMPONENT mqt-core-qdmi-spank)
