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

if(LLVM_ENABLE_PLUGINS)
  add_llvm_library(mqt-catalyst-plugin MODULE mqt-plugin.cpp LINK_LIBS MLIRMQTOpt
                   MLIRMQTOptTransforms)
else()
  # `MODULE` targets are ignored by the scikit-build-core. Hence, we build a shared library instead
  # and set the properties to emulate the behavior of `MODULE`.
  add_llvm_library(mqt-catalyst-plugin mqt-plugin.cpp LINK_LIBS MLIRMQTOpt MLIRMQTOptTransforms)
  # set the properties to emulate the behavior of `MODULE`
  set_target_properties(mqt-catalyst-plugin PROPERTIES OUTPUT_NAME "mqt-catalyst-plugin" PREFIX "")
endif()

# add MQT alias
add_library(MQT::CatalystPlugin ALIAS mqt-catalyst-plugin)
