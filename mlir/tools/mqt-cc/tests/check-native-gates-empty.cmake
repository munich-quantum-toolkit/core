# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(expected "--native-gates must not be empty")

foreach(native_gates_arg IN ITEMS "--native-gates=" "--native-gates=   ")
  execute_process(
    COMMAND "${MQT_CC}" "${INPUT}" --emit=qco-optimized ${native_gates_arg}
    RESULT_VARIABLE result
    ERROR_VARIABLE error)

  if(result EQUAL 0)
    message(FATAL_ERROR "mqt-cc accepted empty ${native_gates_arg} with --emit=qco-optimized")
  endif()

  string(FIND "${error}" "${expected}" diagnostic_position)
  if(diagnostic_position EQUAL -1)
    message(
      FATAL_ERROR "mqt-cc did not emit the expected diagnostic for ${native_gates_arg}:\n${error}")
  endif()
endforeach()
