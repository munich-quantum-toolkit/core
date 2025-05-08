# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MQTOpt)
add_public_tablegen_target(MLIRMQTOptTransformsIncGen)
add_mlir_doc(MQTOptPasses MLIRMQTOptPasses Passes/ -gen-pass-doc)
