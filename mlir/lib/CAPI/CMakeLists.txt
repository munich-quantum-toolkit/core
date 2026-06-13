# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# The C-API library backs the Python bindings. It is a plain static library (rather than
# add_mlir_public_c_api_library) so that it does not eagerly require its linked MLIR libraries to be
# defined targets at configure time, which is not guaranteed across all supported MLIR toolchains.
add_library(MQTCoreCAPI STATIC Registration.cpp)

# Export the C-API symbols from the object files.
target_compile_definitions(MQTCoreCAPI PRIVATE MLIR_CAPI_BUILDING_LIBRARY=1)

# The TableGen'd pass/conversion headers must be generated before compiling. Each dependency is
# added only if the target is already defined, since the order in which targets become visible
# differs across MLIR toolchains. The PUBLIC link dependencies below additionally enforce the
# correct build order once all targets are resolved.
foreach(
  _mqt_capi_dep IN
  ITEMS QCToQCOIncGen
        QCOToQCIncGen
        QCToQIRIncGen
        JeffToQCOIncGen
        QCOToJeffIncGen
        MLIRQCTransformsIncGen
        MLIRQCOTransformsIncGen
        MLIRQIRTransformsIncGen
        MLIRQTensorTransformsIncGen)
  if(TARGET ${_mqt_capi_dep})
    add_dependencies(MQTCoreCAPI ${_mqt_capi_dep})
  endif()
endforeach()

target_link_libraries(
  MQTCoreCAPI
  PUBLIC # Dialects
         MLIRQCDialect
         MLIRQCODialect
         MLIRQTensorDialect
         MLIRJeff
         # Conversions
         MLIRQCToQCO
         MLIRQCOToQC
         MLIRQCToQIR
         MLIRJeffToQCO
         MLIRQCOToJeff
         # Transformations
         MLIRQCTransforms
         MLIRQCOTransforms
         MLIRQIRTransforms
         MLIRQTensorTransforms
         # OpenQASM 3 import
         MLIRQCTranslation
         MQT::CoreQASM
         MQT::CoreIR
         # Upstream
         MLIRCAPIIR
         MLIRIR
         MLIRPass
         MLIRTransforms
         MLIRArithDialect
         MLIRControlFlowDialect
         MLIRFuncDialect
         MLIRLLVMDialect
         MLIRMemRefDialect
         MLIRSCFDialect)

mqt_mlir_target_use_project_options(MQTCoreCAPI)
