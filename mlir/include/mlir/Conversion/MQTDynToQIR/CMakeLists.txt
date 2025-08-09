# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTDynToQIR.td)
mlir_tablegen(MQTDynToQIR.h.inc -gen-pass-decls -name MQTDynToQIR)
add_public_tablegen_target(MQTDynToQIRIncGen)

add_mlir_doc(MQTDynToQIR MLIRMQTDynToQIR Conversions/ -gen-pass-doc)
