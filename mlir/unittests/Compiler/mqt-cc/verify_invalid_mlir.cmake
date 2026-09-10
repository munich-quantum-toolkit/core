# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(require_failure description expected)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    ERROR_VARIABLE error)
  if(NOT result EQUAL 1 OR NOT error MATCHES "${expected}")
    message(FATAL_ERROR "${description} did not fail as expected:\n${error}")
  endif()
endfunction()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(input_file "${OUTPUT_DIR}/invalid.mlir")
file(WRITE "${input_file}" "module {\n")

require_failure("invalid MLIR" "expected operation name" "${MQT_CC}" "${input_file}")
require_failure("nonlinear QCO" "expected linear QCO value to have exactly one use" "${MQT_CC}"
                "${NONLINEAR_QCO_INPUT}" "--emit=qco")

file(
  WRITE "${input_file}"
  [=[
module {
  func.func @main(%input: f64) {
    %q = qc.alloc : !qc.qubit
    %infinity = arith.constant 0x7FF0000000000000 : f64
    %theta = arith.addf %input, %infinity : f64
    qc.rx(%theta) %q : !qc.qubit
    qc.dealloc %q : !qc.qubit
    return
  }
}
]=])
require_failure("non-finite QC import" "constant parameter expression at index 0 must be finite"
                "${MQT_CC}" "${input_file}" "--emit=qc-import")
