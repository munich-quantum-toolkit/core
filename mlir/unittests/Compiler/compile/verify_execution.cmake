# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(run_ok)
  execute_process(
    COMMAND "${MQT_CORE_CLI}" ${ARGN}
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "${ARGN}: ${status}\n${output}${error}")
  endif()
  set(result "${output}" PARENT_SCOPE)
endfunction()

function(run_bad diagnostic)
  execute_process(
    COMMAND "${MQT_CORE_CLI}" ${ARGN}
    RESULT_VARIABLE status
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT status EQUAL 1 OR NOT "${error}" MATCHES "${diagnostic}")
    message(FATAL_ERROR "${ARGN}: expected '${diagnostic}', got ${status}\n${output}${error}")
  endif()
endfunction()

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
set(source "${OUTPUT_DIR}/readout.qasm")
file(WRITE "${source}"
     "OPENQASM 3; qubit[2] q; bit[3] c; x q[0]; c[2]=measure q[0]; c[0]=measure q[0]; c[1]=measure q[1];")
run_ok()
run_ok(bench list)
run_ok(--help)
run_ok(--version)
get_filename_component(binary_dir "${MQT_CORE_CLI}" DIRECTORY)
run_ok(compile --qdmi-list-devices
       "--qdmi-manifest=${binary_dir}/mqt-core-qdmi-ddsim-device.qdmi.json")
if(NOT result MATCHES "mqt.ddsim.default")
  message(FATAL_ERROR "Registered DDSIM device is missing: ${result}")
endif()
foreach(option IN ITEMS --include-dir --cmake-dir)
  execute_process(
    COMMAND "${MQT_CORE_CLI}" "${option}"
    RESULT_VARIABLE status
    OUTPUT_VARIABLE directory
    ERROR_VARIABLE error OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(status EQUAL 0)
    if(NOT IS_DIRECTORY "${directory}")
      message(FATAL_ERROR "${option} returned a missing directory: ${directory}")
    endif()
  elseif(NOT error MATCHES "Installation directory not found")
    message(FATAL_ERROR "${option}: ${error}")
  endif()
endforeach()
run_ok(simulate "${source}" --shots=16 --seed=42)
string(JSON count GET "${result}" counts 101)
if(NOT count EQUAL 16)
  message(FATAL_ERROR "Logical readout was not restored: ${result}")
endif()

# Exercise each source checkpoint and both QIR profiles and encodings.
file(WRITE "${source}" "OPENQASM 3; qubit[2] q; bit[2] c; x q[0]; c=measure q;")
foreach(format IN ITEMS qc jeff qir-base qir-adaptive)
  if(format STREQUAL "qc")
    set(extensions mlir)
  elseif(format STREQUAL "jeff")
    set(extensions jeff)
  else()
    set(extensions ll bc)
  endif()
  foreach(extension IN LISTS extensions)
    set(program "${OUTPUT_DIR}/${format}.${extension}")
    run_ok(compile "${source}" "--emit=${format}" -o "${program}")
    run_ok(run "${program}" --qdmi-device=mqt.ddsim.default --shots=16)
    string(JSON width LENGTH "${result}" counts)
    if(NOT width EQUAL 1)
      message(FATAL_ERROR "Expected deterministic counts: ${result}")
    endif()
    string(JSON bits MEMBER "${result}" counts 0)
    string(JSON count GET "${result}" counts "${bits}")
    if(NOT count EQUAL 16 OR NOT bits STREQUAL "01")
      message(FATAL_ERROR "Incorrect counts: ${result}")
    endif()
  endforeach()
endforeach()

run_bad("requires --qdmi-device" run "${source}")
run_bad("require DDSIM" simulate "${source}" --qdmi-device=other)
run_bad("require DDSIM" run "${source}" --qdmi-device=other --seed=1)
run_bad("must be positive" simulate "${source}" --shots=0)
run_bad("[Nn]o such file|cannot find|not find" simulate "${OUTPUT_DIR}/missing.ll")
file(WRITE "${OUTPUT_DIR}/invalid.qasm" "not OpenQASM")
run_bad("." simulate "${OUTPUT_DIR}/invalid.qasm")
foreach(extension IN ITEMS json qpy)
  file(WRITE "${OUTPUT_DIR}/unsupported.${extension}" "{}")
  run_bad("." simulate "${OUTPUT_DIR}/unsupported.${extension}")
endforeach()
foreach(
  ir
  IN
  ITEMS
  "not LLVM IR"
  "define void @main() { ret void }"
  "define void @main() \"entry_point\" { ret void }"
  "define void @main() \"entry_point\" \"qir_profiles\"=\"invalid\" { ret void }")
  file(WRITE "${OUTPUT_DIR}/invalid.ll" "${ir}")
  run_bad("." simulate "${OUTPUT_DIR}/invalid.ll")
endforeach()
