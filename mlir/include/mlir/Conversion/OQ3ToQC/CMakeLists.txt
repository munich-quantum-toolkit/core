# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(LLVM_TARGET_DEFINITIONS OQ3ToQC.td)
mlir_tablegen(OQ3ToQC.h.inc -gen-pass-decls -name OQ3ToQC)
add_public_tablegen_target(OQ3ToQCIncGen)

add_mlir_doc(OQ3ToQC OQ3ToQC Conversions/ -gen-pass-doc)
