# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(run_compiler expected_result)
  execute_process(
    COMMAND "${MQT_CC}" ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT "${result}" STREQUAL "${expected_result}")
    message(FATAL_ERROR "mqt-cc ${ARGN}: expected ${expected_result}, got ${result}:\n${error}")
  endif()
  set(output
      "${output}"
      PARENT_SCOPE)
  set(error
      "${error}"
      PARENT_SCOPE)
endfunction()

function(reject_compiler expected_diagnostic)
  run_compiler(1 ${ARGN})
  if(NOT error MATCHES "${expected_diagnostic}" OR NOT output STREQUAL "")
    message(FATAL_ERROR "Expected '${expected_diagnostic}' without output:\n${error}\n${output}")
  endif()
endfunction()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(reduced "${OUTPUT_DIR}/reduced.mlir")
file(
  WRITE "${reduced}"
  [=[
module {
  func.func @f(%q: !qco.qubit) -> !qco.qubit {
    %h0 = qco.h %q : !qco.qubit -> !qco.qubit
    %h1 = qco.h %h0 : !qco.qubit -> !qco.qubit
    return %h1 : !qco.qubit
  }
}
]=])

# An empty isolated pipeline preserves both gates, including with generic printing.
run_compiler(0 "${reduced}" --run-pipeline "--pass-pipeline=builtin.module()"
             --mlir-print-op-generic)
string(REGEX MATCHALL "\"qco.h\"" gates "${output}")
list(LENGTH gates gate_count)
if(NOT gate_count EQUAL 2)
  message(FATAL_ERROR "Isolated pipeline changed the input: ${output}")
endif()

foreach(option IN ITEMS pass-pipeline passes)
  run_compiler(0 "${reduced}" --run-pipeline "--${option}=builtin.module(canonicalize,cse)")
  if(output MATCHES "qco.h")
    message(FATAL_ERROR "${option} did not run the registered cleanup passes: ${output}")
  endif()
endforeach()

# Both pipeline routes require registered passes and a module anchor.
foreach(mode IN ITEMS --run-pipeline --emit=qco-optimized)
  reject_compiler("does not refer to a registered pass" "${reduced}" "${mode}"
                  "--pass-pipeline=builtin.module(not-a-pass)")
  reject_compiler("must be anchored on builtin.module" "${reduced}" "${mode}"
                  "--pass-pipeline=func.func(cse)")
  reject_compiler(
    "without disabling multi-threading" "${reduced}" "${mode}" "--pass-pipeline=builtin.module()"
    --mlir-print-ir-module-scope --mlir-disable-threading=false)
  run_compiler(0 "${reduced}" "${mode}" "--pass-pipeline=builtin.module()"
               --mlir-print-ir-module-scope --mlir-disable-threading)
endforeach()

# MLIR verification alone does not enforce QCO linearity.
reject_compiler("expected linear QCO value to have exactly one use" "${NONLINEAR_QCO_INPUT}"
                --run-pipeline "--pass-pipeline=builtin.module()")

run_compiler(0 "${reduced}" --run-pipeline "--pass-pipeline=builtin.module(hadamard-lifting)"
             --mlir-print-ir-before-all --mlir-disable-threading)
string(REGEX MATCHALL "IR Dump Before" dumps "${error}")
list(LENGTH dumps dump_count)
if(NOT dump_count EQUAL 1 OR NOT error MATCHES "Before HadamardLifting")
  message(FATAL_ERROR "Isolated pipeline ran unexpected stages: ${error}")
endif()

# Normal compilation retains cleanup on both sides of the custom pipeline.
run_compiler(0 "${reduced}" --emit=qco-optimized "--pass-pipeline=builtin.module(hadamard-lifting)"
             --mlir-print-ir-before-all --mlir-disable-threading)
string(FIND "${error}" "Before Canonicalizer" before_cleanup)
string(FIND "${error}" "Before HadamardLifting" custom_pass)
string(FIND "${error}" "Before Canonicalizer" after_cleanup REVERSE)
if(before_cleanup LESS 0
   OR NOT before_cleanup LESS custom_pass
   OR NOT custom_pass LESS after_cleanup)
  message(FATAL_ERROR "Compiler preparation/cleanup was lost: ${error}")
endif()

run_compiler(1 "${reduced}" --measurement-lifting)
if(NOT error MATCHES "Unknown command line argument")
  message(FATAL_ERROR "Invalid individual pass flag was not diagnosed: ${error}")
endif()
run_compiler(1 "${reduced}" --run-pipeline "--pass-pipeline=builtin.module()" --emit=qco)
run_compiler(1 "${QASM_INPUT}" --run-pipeline "--pass-pipeline=builtin.module()")
run_compiler(1 "${reduced}" --run-reproducer)
if(NOT error MATCHES "requires a recorded")
  message(FATAL_ERROR "Missing reproducer configuration was not diagnosed: ${error}")
endif()

set(unsupported "${OUTPUT_DIR}/unsupported.mlir")
file(
  WRITE "${unsupported}"
  [=[
module {
  func.func @f() {
    cf.br ^next
  ^next:
    return
  }
}
]=])
set(reproducer "${OUTPUT_DIR}/failure.mlir")
file(REMOVE "${reproducer}")
run_compiler(
  1
  "${unsupported}"
  --mlir-print-ir-before-all
  --mlir-print-ir-after-failure
  --mlir-print-op-on-diagnostic
  --mlir-print-stacktrace-on-diagnostic
  --mlir-disable-threading
  "--mlir-pass-pipeline-crash-reproducer=${reproducer}")
if(NOT EXISTS "${reproducer}"
   OR NOT error MATCHES "After QCToQCO Failed"
   OR NOT error MATCHES "does not support unstructured control flow"
   OR NOT error MATCHES "note: diagnostic emitted with trace:"
   OR NOT error MATCHES "note: see current operation"
   OR NOT error MATCHES "cf[.]br .next")
  message(
    FATAL_ERROR "Failure diagnostics, source excerpts, notes, or reproducer missing: ${error}")
endif()

# Retain stdin's source buffer through the later conversion failure as well.
execute_process(
  COMMAND "${MQT_CC}" - --input-format=mlir --mlir-print-op-on-diagnostic
  INPUT_FILE "${unsupported}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE output
  ERROR_VARIABLE error)
if(NOT result EQUAL 1
   OR NOT error MATCHES "cf[.]br .next"
   OR NOT error MATCHES "note: see current operation")
  message(FATAL_ERROR "stdin source diagnostics were lost: ${error}")
endif()

# Reloading the saved pipeline must reproduce the same conversion failure.
run_compiler(1 "${reproducer}" --run-reproducer --mlir-print-ir-before-all)
if(NOT error MATCHES "Before QCToQCO" OR NOT error MATCHES
                                         "does not support unstructured control flow")
  message(FATAL_ERROR "Saved pipeline was not replayed: ${error}")
endif()
run_compiler(1 "${unsupported}" --run-pipeline "--pass-pipeline=builtin.module(qc-to-qco)")

# Later QIR failures must replay even when cleanup passes follow the failing pass.
foreach(profile IN ITEMS qir-base qir-adaptive)
  set(qir_reproducer "${OUTPUT_DIR}/${profile}-failure.mlir")
  file(REMOVE "${qir_reproducer}")
  reject_compiler(
    "no main function with mqt.entry_point found" "${reduced}" "--emit=${profile}"
    --mlir-disable-threading "--mlir-pass-pipeline-crash-reproducer=${qir_reproducer}")
  reject_compiler("no main function with mqt.entry_point found" "${qir_reproducer}"
                  --run-reproducer)
endforeach()

# Register the named shrink and QIR cleanup passes for isolated pipelines too.
run_compiler(
  0 "${reduced}" --run-pipeline
  "--pass-pipeline=builtin.module(qtensor-shrink-to-fit,qc-shrink-qubit-registers,qir-cleanup)")
reject_compiler("QIR metadata attachment requires exactly one entry point" "${reduced}"
                --run-pipeline "--pass-pipeline=builtin.module(set-qir-attributes-and-metadata)")

# Reproducer verification settings are intentional, including disabled verification.
set(unverified "${OUTPUT_DIR}/unverified.mlir")
file(
  WRITE "${unverified}"
  [=[
module {
  func.func @f() {
    func.call @missing() : () -> ()
    return
  }
}
{-#
external_resources: {
  mlir_reproducer: {
    pipeline: "builtin.module(cse)",
    disable_threading: true,
    verify_each: false
  }
}
#-}
]=])
run_compiler(1 "${unverified}" --run-pipeline "--pass-pipeline=builtin.module(cse)")
run_compiler(0 "${unverified}" --run-reproducer)
if(NOT output MATCHES "func.call" OR NOT output MATCHES "@missing")
  message(FATAL_ERROR "Reproducer did not preserve the unchecked input: ${output}")
endif()

# Invalid recorded pipelines must fail before executing or writing the input.
file(READ "${unverified}" reproducer_source)
set(invalid_reproducer "${OUTPUT_DIR}/invalid-reproducer.mlir")
string(REPLACE "builtin.module(cse)" "builtin.module(not-a-pass)" invalid_source
               "${reproducer_source}")
file(WRITE "${invalid_reproducer}" "${invalid_source}")
reject_compiler("does not refer to a registered pass" "${invalid_reproducer}" --run-reproducer)
foreach(pipeline IN ITEMS "builtin.module()" "func.func(cse)")
  string(REPLACE "builtin.module(cse)" "${pipeline}" invalid_source "${reproducer_source}")
  file(WRITE "${invalid_reproducer}" "${invalid_source}")
  reject_compiler("requires a recorded, non-empty builtin.module pipeline" "${invalid_reproducer}"
                  --run-reproducer)
endforeach()

# The initial jeff conversion is part of the requested instrumentation.
set(jeff "${OUTPUT_DIR}/input.jeff")
run_compiler(0 "${QASM_INPUT}" --emit=jeff -o "${jeff}")
reject_compiler("without disabling multi-threading" "${jeff}" --emit=qco
                --mlir-print-ir-module-scope --mlir-disable-threading=false)
run_compiler(0 "${jeff}" --emit=qco --mlir-print-ir-before-all --mlir-print-ir-module-scope
             --mlir-disable-threading)
if(NOT error MATCHES "Before JeffToQCO" OR NOT output MATCHES "qco[.]")
  message(FATAL_ERROR "Initial jeff conversion was not instrumented: ${error}")
endif()

# Reusable calls must be inlined before a custom QCO pipeline on the QIR route.
set(reusable "${OUTPUT_DIR}/reusable.mlir")
file(
  WRITE "${reusable}"
  [=[
module {
  func.func private @flip(%q: !qco.qubit) -> !qco.qubit attributes {mqt.unitary} {
    %out = qco.x %q : !qco.qubit -> !qco.qubit
    return %out : !qco.qubit
  }
  func.func @main() attributes {mqt.entry_point} {
    %q = qco.alloc : !qco.qubit
    %out = qco.call @flip(%q) : (!qco.qubit) -> !qco.qubit
    qco.sink %out : !qco.qubit
    return
  }
}
]=])
run_compiler(0 "${reusable}" --emit=qir-adaptive "--pass-pipeline=builtin.module(hadamard-lifting)"
             --mlir-print-ir-before-all --mlir-disable-threading)
string(FIND "${error}" "Before Inliner" inliner)
string(FIND "${error}" "Before HadamardLifting" custom_pass)
if(inliner LESS 0
   OR NOT inliner LESS custom_pass
   OR NOT output MATCHES "define void @main")
  message(FATAL_ERROR "Custom pipeline discarded required inlining: ${error}")
endif()
