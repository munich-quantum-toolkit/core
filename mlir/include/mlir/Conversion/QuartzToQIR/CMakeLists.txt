# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS QuartzToQIR.td)
mlir_tablegen(QuartzToQIR.h.inc -gen-pass-decls -name QuartzToQIR)
add_public_tablegen_target(QuartzToQIRIncGen)

add_mlir_doc(QuartzToQIR MLIRQuartzToQIR Conversions/ -gen-pass-doc)
