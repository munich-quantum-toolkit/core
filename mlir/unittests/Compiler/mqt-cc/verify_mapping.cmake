# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

foreach(option mapping-seed mapping-trials)
  execute_process(
    COMMAND "${MQT_CC}" "--${option}=3"
    RESULT_VARIABLE result
    ERROR_VARIABLE error
    OUTPUT_QUIET)
  if(result EQUAL 0 OR NOT error MATCHES "Mapping controls require --qdmi-device")
    message(FATAL_ERROR "${option} did not reject a missing target: ${error}")
  endif()
endforeach()

execute_process(
  COMMAND "${MQT_CC}" --qdmi-device mqt.ddsim.default --mapping-trials=0
  RESULT_VARIABLE result
  ERROR_VARIABLE error
  OUTPUT_QUIET)
if(result EQUAL 0 OR NOT error MATCHES "mapping-trials must be greater than zero")
  message(FATAL_ERROR "Zero mapping trials were not rejected: ${error}")
endif()
