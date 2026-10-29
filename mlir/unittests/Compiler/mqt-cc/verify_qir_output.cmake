# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(run_command description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${description} failed with exit code ${result}:\n${output}${error}")
  endif()
endfunction()

function(run_command_expect_failure description expected_error)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(result EQUAL 0)
    message(FATAL_ERROR "${description} unexpectedly succeeded")
  endif()
  string(FIND "${output}${error}" "${expected_error}" error_position)
  if(error_position EQUAL -1)
    message(FATAL_ERROR "${description} did not report '${expected_error}':\n${output}${error}")
  endif()
endfunction()

function(require_profile filename expected_profile)
  file(READ "${filename}" llvm_ir)
  string(FIND "${llvm_ir}" "\"qir_profiles\"=\"${expected_profile}\"" profile_position)
  if(profile_position EQUAL -1)
    message(FATAL_ERROR "${filename} does not declare QIR profile ${expected_profile}")
  endif()
endfunction()

function(require_textual_llvm_ir filename)
  file(READ "${filename}" llvm_ir_header LIMIT 10)
  string(FIND "${llvm_ir_header}" "; ModuleID" module_id_position)
  if(NOT module_id_position EQUAL 0)
    message(FATAL_ERROR "${filename} is not textual LLVM IR")
  endif()
endfunction()

function(require_bitcode filename)
  file(READ "${filename}" bitcode_magic OFFSET 0 LIMIT 4 HEX)
  string(TOLOWER "${bitcode_magic}" bitcode_magic)
  if(NOT bitcode_magic STREQUAL "4243c0de")
    message(FATAL_ERROR "${filename} does not start with the LLVM bitcode magic")
  endif()
endfunction()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

set(binary_payload_specification
    "#mqt.payload_spec<format = <id = \"qir\", version = \"2.1.0\", profile = \"base\", encoding = binary>, capabilities = [], optional_capabilities_known = false>"
)
set(text_payload_specification
    "#mqt.payload_spec<format = <id = \"qir\", version = \"2.1.0\", profile = \"base\", encoding = text>, capabilities = [], optional_capabilities_known = false>"
)
run_command_expect_failure("mqt-cc target without payload specification"
                           "--qdmi-device or --target and --payload-spec must be provided together"
                           "${MQT_CC}" "${INPUT_FILE}" "--qdmi-device=mqt.sc.iqm.garnet")
run_command_expect_failure(
  "mqt-cc invalid payload specification"
  "--payload-spec must be a valid #mqt.payload_spec attribute" "${MQT_CC}" "${INPUT_FILE}"
  "--qdmi-device=mqt.sc.iqm.garnet" "--payload-spec=#mqt.payload_spec<>"
)
run_command_expect_failure(
  "mqt-cc target with explicit output"
  "Only --emit=qco-optimized can be combined with target compilation"
  "${MQT_CC}"
  "${INPUT_FILE}"
  "--qdmi-device=mqt.sc.iqm.garnet"
  "--payload-spec=${binary_payload_specification}"
  "--emit=qir-base")

execute_process(
  COMMAND
    "${MQT_CC}"
    "${INPUT_FILE}"
    "--qdmi-device=mqt.sc.iqm.garnet"
    "--payload-spec=${binary_payload_specification}"
    --emit=qco-optimized
    -o
    -
  RESULT_VARIABLE result
  OUTPUT_VARIABLE mapped_qco
  ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT mapped_qco MATCHES "mqt\\.layout =")
  message(FATAL_ERROR "mqt-cc did not emit mapped QCO with layout metadata: ${error}")
endif()

set(target_bitcode_file "${OUTPUT_DIR}/target-binary.ll")
set(target_disassembled_file "${OUTPUT_DIR}/target-binary-disassembled.ll")
run_command(
  "mqt-cc binary payload target compilation"
  "${MQT_CC}"
  "${INPUT_FILE}"
  "--qdmi-device=mqt.sc.iqm.garnet"
  "--payload-spec=${binary_payload_specification}"
  -o
  "${target_bitcode_file}")
require_bitcode("${target_bitcode_file}")
run_command("llvm-dis target bitcode validation" "${LLVM_DIS}" "${target_bitcode_file}" -o
            "${target_disassembled_file}")
require_profile("${target_disassembled_file}" "base_profile")

set(target_text_file "${OUTPUT_DIR}/target-text.bc")
set(target_assembled_file "${OUTPUT_DIR}/target-text-assembled.bc")
run_command(
  "mqt-cc text payload target compilation"
  "${MQT_CC}"
  "${INPUT_FILE}"
  "--qdmi-device=mqt.sc.iqm.garnet"
  "--payload-spec=${text_payload_specification}"
  -o
  "${target_text_file}")
require_textual_llvm_ir("${target_text_file}")
run_command("llvm-as target text validation" "${LLVM_AS}" "${target_text_file}" -o
            "${target_assembled_file}")
require_profile("${target_text_file}" "base_profile")

set(adaptive_target
    "#mqt.compilation_target<sites = [<id = 0>, <id = 1>], connectivity = all_to_all, couplings = [], native_operations = explicit, operations = [<name = \"u\", arity = <kind = fixed, value = 1>, num_parameters = 3, site_tuples = []>, <name = \"cz\", arity = <kind = fixed, value = 2>, num_parameters = 0, site_tuples = []>, <name = \"measure\", arity = <kind = fixed, value = 1>, num_parameters = 0, site_tuples = []>, <name = \"reset\", arity = <kind = fixed, value = 1>, num_parameters = 0, site_tuples = []>]>"
)
set(adaptive_payload
    "#mqt.payload_spec<format = <id = \"qir\", version = \">=2.0.0,<3.0.0\", profile = \"adaptive\", encoding = text>, capabilities = [<id = \"forward-branching\", value = 0, constraints = []>, <id = \"counted-iteration\", value = 0, constraints = []>, <id = \"qir.dynamic-result-management\", value = 0, constraints = []>, <id = \"qir.arrays\", value = 0, constraints = []>, <id = \"qir.int-computations\", value = 0, constraints = []>], optional_capabilities_known = false>"
)
string(REPLACE "<id = \"qir.arrays\", value = 0, constraints = []>, " "" unsupported_payload
       "${adaptive_payload}")
run_command_expect_failure(
  "mqt-cc rejects unsupported result arrays"
  "requires capability 'qir.arrays'"
  "${MQT_CC}"
  "${INPUT_FILE}"
  "--target=${adaptive_target}"
  "--payload-spec=${unsupported_payload}"
  -o
  "${OUTPUT_DIR}/unsupported.ll")

file(WRITE "${OUTPUT_DIR}/bounded.qasm"
     "OPENQASM 3.1;\ninclude \"stdgates.inc\";\nqubit[2] q;\nfor int i in [0:2] { x q[0]; }\n")
string(REPLACE "<id = \"counted-iteration\", value = 0, constraints = []>"
       "<id = \"counted-iteration\", value = 0, constraints = [<id = \"max-iteration-count\", value = 3>]>"
       bounded_payload "${adaptive_payload}")
run_command(
  "mqt-cc accepts a proved counted-loop bound"
  "${MQT_CC}"
  "${OUTPUT_DIR}/bounded.qasm"
  "--target=${adaptive_target}"
  "--payload-spec=${bounded_payload}"
  -o
  "${OUTPUT_DIR}/bounded.ll")
file(READ "${OUTPUT_DIR}/bounded.ll" bounded_qir)
if(NOT bounded_qir MATCHES "backwards_branching\", i2 1")
  message(FATAL_ERROR "The bounded loop was not retained in QIR")
endif()

foreach(profile IN ITEMS base adaptive)
  set(expected_profile "${profile}_profile")
  set(text_file "${OUTPUT_DIR}/${profile}.ll")
  set(text_bitcode_file "${OUTPUT_DIR}/${profile}-from-text.bc")
  set(bitcode_file "${OUTPUT_DIR}/${profile}.bc")
  set(disassembled_file "${OUTPUT_DIR}/${profile}-from-bitcode.ll")

  run_command("mqt-cc ${profile} textual QIR generation" "${MQT_CC}" "${INPUT_FILE}"
              "--emit=qir-${profile}" -o "${text_file}")
  require_textual_llvm_ir("${text_file}")
  run_command("llvm-as ${profile} textual QIR validation" "${LLVM_AS}" "${text_file}" -o
              "${text_bitcode_file}")
  require_profile("${text_file}" "${expected_profile}")

  run_command("mqt-cc ${profile} bitcode generation" "${MQT_CC}" "${INPUT_FILE}"
              "--emit=qir-${profile}" -o "${bitcode_file}")
  require_bitcode("${bitcode_file}")
  run_command("llvm-dis ${profile} bitcode validation" "${LLVM_DIS}" "${bitcode_file}" -o
              "${disassembled_file}")
  require_profile("${disassembled_file}" "${expected_profile}")
endforeach()
