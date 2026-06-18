# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_mlir_tool(mqt-cc-translate mqt-cc-translate.cpp DEPENDS MQTCCTranslation SUPPORT_PLUGINS)
target_link_libraries(mqt-cc-translate PRIVATE MQTCCTranslation)

mqt_mlir_target_use_project_options(mqt-cc-translate)
llvm_update_compile_flags(mqt-cc-translate)
mlir_check_all_link_libraries(mqt-cc-translate)
