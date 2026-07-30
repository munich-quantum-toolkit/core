# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT DEFINED EXPECTED)
  message(FATAL_ERROR "EXPECTED must be set")
endif()

execute_process(
  COMMAND ${MQT_CC} ${INPUT} ${ARGS}
  RESULT_VARIABLE result
  ERROR_VARIABLE error)

if(result EQUAL 0)
  message(FATAL_ERROR "mqt-cc unexpectedly succeeded:\n${error}")
endif()

string(FIND "${error}" "${EXPECTED}" diagnostic_position)
if(diagnostic_position EQUAL -1)
  message(FATAL_ERROR "mqt-cc did not emit the expected diagnostic '${EXPECTED}':\n${error}")
endif()
