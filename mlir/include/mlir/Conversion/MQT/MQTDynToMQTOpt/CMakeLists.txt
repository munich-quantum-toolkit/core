# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTDynToMQTOpt.td)
mlir_tablegen(MQTDynToMQTOpt.h.inc -gen-pass-decls -name MQTDynToMQTOpt)
add_public_tablegen_target(MQTDynToMQTOptIncGen)

add_mlir_doc(MQTDynToMQTOpt MQTDynToMQTOpt ./ -gen-pass-doc)
