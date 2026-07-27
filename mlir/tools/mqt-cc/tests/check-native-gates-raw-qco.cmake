# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

execute_process(
  COMMAND "${MQT_CC}" "${INPUT}" --emit=qco --native-gates=u,cx
  RESULT_VARIABLE result
  ERROR_VARIABLE error)

if(result EQUAL 0)
  message(FATAL_ERROR "mqt-cc accepted --native-gates with raw QCO output")
endif()

set(expected "--native-gates requires an output that passes through QCO optimization")
string(FIND "${error}" "${expected}" diagnostic_position)
if(diagnostic_position EQUAL -1)
  message(FATAL_ERROR "mqt-cc did not emit the expected diagnostic:\n${error}")
endif()
