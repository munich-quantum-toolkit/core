# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_compile_options(-fexceptions)

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

add_llvm_library(
  mqt-catalyst-plugin
  MODULE
  mqt-plugin.cpp
  LINK_LIBS
  MLIRMQTOpt
  MLIRMQTOptTransforms
  CatalystQuantumToMQTOpt
  MQTOptToCatalystQuantum)

# set required C++ standard
target_compile_features(mqt-catalyst-plugin PUBLIC cxx_std_17)

# Install directive for scikit-build-core
install(
  TARGETS mqt-catalyst-plugin
  DESTINATION lib
  COMPONENT mqt-catalyst_Plugin)

# Install the Python stub files in editable mode for better IDE support
if(SKBUILD_STATE STREQUAL "editable")
  file(GLOB IR_PYI_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../python/mqt/catalyst/*.pyi)
  install(
    FILES ${IR_PYI_FILES}
    DESTINATION .
    COMPONENT mqt-catalyst_Plugin)
endif()

# add MQT alias
add_library(MQT::CatalystPlugin ALIAS mqt-catalyst-plugin)
