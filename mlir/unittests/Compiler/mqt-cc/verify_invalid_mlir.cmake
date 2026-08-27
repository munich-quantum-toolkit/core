# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(input_file "${OUTPUT_DIR}/invalid.mlir")
file(WRITE "${input_file}" "module {\n")

execute_process(
  COMMAND "${MQT_CC}" "${input_file}"
  RESULT_VARIABLE result
  ERROR_VARIABLE error)

if(NOT result EQUAL 1)
  message(FATAL_ERROR "mqt-cc returned ${result} for invalid MLIR:\n${error}")
endif()
if(NOT error MATCHES "expected operation name")
  message(FATAL_ERROR "mqt-cc did not report the invalid MLIR:\n${error}")
endif()
