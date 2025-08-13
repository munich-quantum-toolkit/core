# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTRefToMQTOpt.td)
mlir_tablegen(MQTRefToMQTOpt.h.inc -gen-pass-decls -name MQTRefToMQTOpt)
add_public_tablegen_target(MQTRefToMQTOptIncGen)

add_mlir_doc(MQTRefToMQTOpt MLIRMQTRefToMQTOpt Conversions/ -gen-pass-doc)
