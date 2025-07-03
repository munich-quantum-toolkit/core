# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS MQTDynToQuake.td)
mlir_tablegen(MQTDynToQuake.h.inc -gen-pass-decls -name MQTDynToQuake)
add_public_tablegen_target(MQTDynToQuakeIncGen)

add_mlir_doc(MQTDynToQuake MQTDynToQuake ./ -gen-pass-doc)
