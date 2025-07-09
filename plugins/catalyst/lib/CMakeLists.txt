# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Q: Why is it add_llvm_library and not add_mlir_library? A: According to Erick from Xanadu, who
# talked with the MLIR team, the reason is probably, that the MLIR plugin uses the same
# infrastructure as the LLVM plugins. Probably due to laziness, the appropriate MLIR macros were not
# adopted yet.
#
# Note: On DLL platforms, the tool that will use this plugin must be linked, see
# https://github.com/llvm/llvm-project/blob/2acecfe65397c162958ab305dc44614ff51e748c/llvm/cmake/modules/AddLLVM.cmake#L770
# and
# https://github.com/llvm/llvm-project/blob/2acecfe65397c162958ab305dc44614ff51e748c/llvm/cmake/modules/AddLLVM.cmake#L517
# This would mean that it is not enough to have the corresponding header at hand, we would need to
# have the catalyst target available to link against it.

add_subdirectory(Conversion)

set(TARGET_NAME mqt-core-catalyst-plugin)

add_llvm_library(
  ${TARGET_NAME}
  MODULE
  mqt-plugin.cpp
  LINK_LIBS
  MLIRMQTOptTransforms
  CatalystQuantumToMQTOpt
  MQTOptToCatalystQuantum)

# set required C++ standard
target_compile_features(mqt-catalyst-plugin PUBLIC cxx_std_17)

# RPATH settings to make sure the MQT Core IR library is found at runtime.
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

# install directive for scikit-build-core
install(
  TARGETS ${TARGET_NAME}
  DESTINATION .
  COMPONENT mqt-core-catalyst_Plugin)
