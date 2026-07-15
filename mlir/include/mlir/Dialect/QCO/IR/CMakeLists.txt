# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect(QCOOps qco)
add_mlir_interface(QCOInterfaces)

add_mlir_doc(QCOOps QCODialect Dialects/ -gen-dialect-doc)
add_mlir_doc(QCOInterfaces QCOInterfaces Dialects/ -gen-op-interface-docs -dialect=qco)

set(LLVM_TARGET_DEFINITIONS QCOAttrDefs.td)
mlir_tablegen(QCOEnums.h.inc -gen-enum-decls)
mlir_tablegen(QCOEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(QCOAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=quantum)
mlir_tablegen(QCOAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=quantum)
add_public_tablegen_target(MLIRQCOAttrDefsIncGen)
