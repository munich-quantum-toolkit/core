# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(QIRCommon)
add_subdirectory(QIRBase)
add_subdirectory(QIRAdaptive)

# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM Copyright (c) 2025 - 2026 Munich
# Quantum Software Company GmbH All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(GLOB CONVERSION_SOURCES *.cpp)

add_mlir_conversion_library(MLIRQCToQIR ${CONVERSION_SOURCES} LINK_LIBS MLIRQCToQIRBase
                            MLIRQCToQIRAdaptive)

mqt_mlir_target_use_project_options(MLIRQCToQCO)
