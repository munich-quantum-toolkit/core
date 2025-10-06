# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QuartzOps.td)
mlir_tablegen(QuartzOps.h.inc -gen-op-decls)
mlir_tablegen(QuartzOps.cpp.inc -gen-op-defs)
mlir_tablegen(QuartzOpsDialect.h.inc -gen-dialect-decls -dialect=quartz)
mlir_tablegen(QuartzOpsDialect.cpp.inc -gen-dialect-defs -dialect=quartz)
mlir_tablegen(QuartzOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=quartz)
mlir_tablegen(QuartzOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=quartz)
add_public_tablegen_target(MLIRQuartzOpsIncGen)

add_mlir_doc(QuartzDialect QuartzDialect Dialects/ -gen-dialect-doc)
add_mlir_doc(QuartzOps QuartzOps Dialects/ -gen-op-doc)
add_mlir_doc(QuartzTypes QuartzTypes Dialects/ -gen-typedef-doc)

set(LLVM_TARGET_DEFINITIONS QuartzInterfaces.td)
mlir_tablegen(QuartzOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(QuartzOpsInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRQuartzInterfacesIncGen)
