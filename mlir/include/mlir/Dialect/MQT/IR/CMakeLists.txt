# Copyright (c) 2026 Chair for Design Automation, TUM
# Copyright (c) 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTDialect.td)
mlir_tablegen(MQTDialect.h.inc -gen-dialect-decls -dialect=mqt)
mlir_tablegen(MQTDialect.cpp.inc -gen-dialect-defs -dialect=mqt)
add_public_tablegen_target(MLIRMQTDialectIncGen)

add_mlir_doc(MQTDialect MQTDialect Dialects/ -gen-dialect-doc -dialect=mqt)
