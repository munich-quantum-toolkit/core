# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_dialect(OQ3Ops oq3)

set(LLVM_TARGET_DEFINITIONS OQ3Attrs.td)
mlir_tablegen(OQ3OpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(OQ3OpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIROQ3EnumsIncGen)

add_mlir_doc(OQ3Ops OQ3Dialect Dialects/ -gen-dialect-doc)
