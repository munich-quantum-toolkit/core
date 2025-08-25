# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QIRToMQTRef.td)
mlir_tablegen(QIRToMQTRef.h.inc -gen-pass-decls -name QIRToMQTRef)
add_public_tablegen_target(QIRToMQTRefIncGen)

add_mlir_doc(QIRToMQTRef MLIRQIRToMQTRef Conversions/ -gen-pass-doc)
