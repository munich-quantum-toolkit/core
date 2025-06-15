# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_compile_options(-fexceptions)

if(APPLE)
  set(BASEPOINT @loader_path)
else()
  set(BASEPOINT $ORIGIN)
endif()

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

# where to find the MQT-core library
list(APPEND CMAKE_INSTALL_RPATH ${BASEPOINT}/../../src/ir)
set_target_properties(
  mqt-catalyst-plugin
  PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}"
             BUILD_RPATH "${CMAKE_INSTALL_RPATH}"
             INSTALL_NAME_DIR "@rpath")

# set required C++ standard
target_compile_features(mqt-catalyst-plugin PUBLIC cxx_std_17)

include(GNUInstallDirs)
# Install directive for scikit-build-core
install(
  TARGETS mqt-catalyst-plugin
  EXPORT mqt-catalyst-plugin-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
          FILE_SET HEADERS
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mqt/catalyst
          COMPONENT mqt-catalyst_Plugin)

# add MQT alias
add_library(MQT::CatalystPlugin ALIAS mqt-catalyst-plugin)
