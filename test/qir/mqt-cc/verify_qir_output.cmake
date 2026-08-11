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

file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

foreach(profile IN ITEMS base adaptive)
  set(target_options)
  if(profile STREQUAL "base")
    list(APPEND target_options "--qdmi-device=mqt.ddsim.default")
  endif()
  set(expected_profile "${profile}_profile")
  set(text_file "${OUTPUT_DIR}/${profile}.ll")
  set(text_bitcode_file "${OUTPUT_DIR}/${profile}-from-text.bc")
  set(bitcode_file "${OUTPUT_DIR}/${profile}.bc")
  set(disassembled_file "${OUTPUT_DIR}/${profile}-from-bitcode.ll")

  run_command(
    "mqt-cc ${profile} textual QIR generation"
    "${MQT_CC}"
    "${INPUT_FILE}"
    "--emit=qir-${profile}"
    ${target_options}
    -o
    "${text_file}")
  require_textual_llvm_ir("${text_file}")
  run_command("llvm-as ${profile} textual QIR validation" "${LLVM_AS}" "${text_file}" -o
              "${text_bitcode_file}")
  require_profile("${text_file}" "${expected_profile}")

  run_command("mqt-cc ${profile} bitcode generation" "${MQT_CC}" "${INPUT_FILE}"
              "--emit=qir-${profile}" -o "${bitcode_file}")
  file(
    READ "${bitcode_file}" bitcode_magic
    OFFSET 0
    LIMIT 4
    HEX)
  string(TOLOWER "${bitcode_magic}" bitcode_magic)
  if(NOT bitcode_magic STREQUAL "4243c0de")
    message(FATAL_ERROR "${bitcode_file} does not start with the LLVM bitcode magic")
  endif()
  run_command("llvm-dis ${profile} bitcode validation" "${LLVM_DIS}" "${bitcode_file}" -o
              "${disassembled_file}")
  require_profile("${disassembled_file}" "${expected_profile}")
endforeach()
