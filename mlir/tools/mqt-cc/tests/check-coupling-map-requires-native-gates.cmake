# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(expected "--coupling-map requires --native-gates")

execute_process(
  COMMAND "${MQT_CC}" "${INPUT}" --emit=qco-optimized --coupling-map=0-1
  RESULT_VARIABLE result
  ERROR_VARIABLE error)

if(result EQUAL 0)
  message(FATAL_ERROR "mqt-cc accepted --coupling-map without --native-gates")
endif()

string(FIND "${error}" "${expected}" diagnostic_position)
if(diagnostic_position EQUAL -1)
  message(FATAL_ERROR "mqt-cc did not emit the expected diagnostic:\n${error}")
endif()
