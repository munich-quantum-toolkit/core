# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS CBitOps.td)
mlir_tablegen(CBitOps.h.inc -gen-op-decls)
mlir_tablegen(CBitOps.cpp.inc -gen-op-defs)
mlir_tablegen(CBitOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=cbit)
mlir_tablegen(CBitOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=cbit)
mlir_tablegen(CBitOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=cbit)
mlir_tablegen(CBitOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=cbit)

set(LLVM_TARGET_DEFINITIONS CBitDialect.td)
mlir_tablegen(CBitOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(CBitOpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(CBitOpsDialect.h.inc -gen-dialect-decls -dialect=cbit)
mlir_tablegen(CBitOpsDialect.cpp.inc -gen-dialect-defs -dialect=cbit)
add_mlir_dialect_tablegen_target(MLIRCBitOpsIncGen)

add_mlir_doc(CBitOps CBitDialect Dialects/ -gen-dialect-doc -dialect=cbit)
