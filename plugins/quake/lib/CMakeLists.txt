# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(Conversion)

add_llvm_library(
  mqt-quake-plugin
  MODULE
  mqtQuakePlugin.cpp
  LINK_LIBS
  MLIRMQTDynTransforms
  MQTDynToQuake
  QuakeToMQTDyn)

# set required C++ standard
target_compile_features(mqt-quake-plugin PUBLIC cxx_std_17)

# RPATH settings to make sure the MQT Core IR library is found at runtime
include(GNUInstallDirs)
list(
  APPEND
  CMAKE_INSTALL_RPATH
  ${BASEPOINT}/../core/${CMAKE_INSTALL_LIBDIR}
  ${BASEPOINT}/../core/lib
  ${BASEPOINT}/../core/lib64
  ${BASEPOINT}/../../core/${CMAKE_INSTALL_LIBDIR}
  ${BASEPOINT}/../../core/lib
  ${BASEPOINT}/../../core/lib64)
