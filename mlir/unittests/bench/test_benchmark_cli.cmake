# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(run_success description output_variable)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${description} failed with exit code ${result}:\n${output}${error}")
  endif()
  set(${output_variable}
      "${output}"
      PARENT_SCOPE)
endfunction()

function(run_failure description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(result EQUAL 0)
    message(FATAL_ERROR "${description} unexpectedly succeeded:\n${output}${error}")
  endif()
endfunction()

if(NOT DEFINED OUTPUT_DIR OR OUTPUT_DIR STREQUAL "")
  message(FATAL_ERROR "OUTPUT_DIR must name the test output directory")
endif()

file(REMOVE_RECURSE "${OUTPUT_DIR}")
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

run_success("top-level help" help_output "${CLI}" --help)
if(help_output MATCHES "cfg-hide-cold-paths")
  message(FATAL_ERROR "help exposed unrelated LLVM options")
endif()

run_success("benchmark listing" list_output "${CLI}" list)
string(JSON benchmark_count LENGTH "${list_output}" benchmarks)
if(NOT benchmark_count EQUAL 9)
  message(FATAL_ERROR "list returned ${benchmark_count} benchmarks instead of 9")
endif()

run_success("multiplexer description" describe_output "${CLI}" describe multiplexer)
string(JSON schema GET "${describe_output}" "$schema")
if(NOT schema STREQUAL "https://json-schema.org/draft/2020-12/schema")
  message(FATAL_ERROR "describe did not return a JSON Schema")
endif()
string(
  JSON
  minimum_qubits
  GET
  "${describe_output}"
  properties
  parameters
  properties
  qubits
  minimum)
if(NOT minimum_qubits EQUAL 2)
  message(FATAL_ERROR "multiplexer schema returned minimum ${minimum_qubits} instead of 2")
endif()

set(instance_specification "${OUTPUT_DIR}/instance-specification.json")
file(WRITE "${instance_specification}"
     "{\"schema_version\":1,\"benchmark\":\"multiplexer\",\"parameters\":{\"qubits\":2}}\n")
set(qc_directory "${OUTPUT_DIR}/qc")
run_success(
  "QC generation"
  generate_output
  "${CLI}"
  generate
  --instance-specification
  "${instance_specification}"
  --format
  qc
  --output
  "${qc_directory}")
string(JSON case_id GET "${generate_output}" case_id)
string(JSON program_path GET "${generate_output}" program_path)
string(JSON manifest_path GET "${generate_output}" manifest_path)
if(NOT EXISTS "${program_path}" OR NOT EXISTS "${manifest_path}")
  message(FATAL_ERROR "generate did not publish both output files")
endif()
get_filename_component(program_name "${program_path}" NAME)
if(NOT program_name MATCHES "^multiplexer-sha256-[0-9a-f]+\\.qc\\.mlir$")
  message(FATAL_ERROR "unexpected program name: ${program_name}")
endif()
get_filename_component(manifest_name "${manifest_path}" NAME)
if(NOT manifest_name MATCHES "^multiplexer-sha256-[0-9a-f]+\\.qc\\.manifest\\.json$")
  message(FATAL_ERROR "unexpected manifest name: ${manifest_name}")
endif()
file(SHA256 "${program_path}" original_program_hash)
file(SHA256 "${manifest_path}" original_manifest_hash)

set(manifest_only_directory "${OUTPUT_DIR}/manifest-only")
file(MAKE_DIRECTORY "${manifest_only_directory}")
set(manifest_only_path "${manifest_only_directory}/${manifest_name}")
set(manifest_only_program_path "${manifest_only_directory}/${program_name}")
file(COPY_FILE "${manifest_path}" "${manifest_only_path}")
run_failure(
  "generation beside an existing manifest"
  "${CLI}"
  generate
  --instance-specification
  "${instance_specification}"
  --format
  qc
  --output
  "${manifest_only_directory}")
file(SHA256 "${manifest_only_path}" retained_manifest_only_hash)
if(EXISTS "${manifest_only_program_path}" OR NOT retained_manifest_only_hash STREQUAL
                                             original_manifest_hash)
  message(FATAL_ERROR "a manifest collision created a program or changed the manifest")
endif()

run_failure(
  "generation beside existing outputs"
  "${CLI}"
  generate
  --instance-specification
  "${instance_specification}"
  --format
  qc
  --output
  "${qc_directory}")
file(SHA256 "${program_path}" retained_program_hash)
file(SHA256 "${manifest_path}" retained_manifest_hash)
if(NOT retained_program_hash STREQUAL original_program_hash OR NOT retained_manifest_hash STREQUAL
                                                               original_manifest_hash)
  message(FATAL_ERROR "a rejected generation changed an existing output")
endif()
file(GLOB qc_temporary_files "${qc_directory}/*.tmp-*")
if(qc_temporary_files)
  message(FATAL_ERROR "successful generation left temporary files")
endif()

set(percent_directory "${OUTPUT_DIR}/literal-%")
run_success(
  "generation in a path containing a percent sign"
  percent_output
  "${CLI}"
  generate
  --instance-specification
  "${instance_specification}"
  --format
  qc
  --output
  "${percent_directory}")
string(JSON percent_program_path GET "${percent_output}" program_path)
string(JSON percent_manifest_path GET "${percent_output}" manifest_path)
if(NOT EXISTS "${percent_program_path}" OR NOT EXISTS "${percent_manifest_path}")
  message(FATAL_ERROR "generation in a path containing a percent sign omitted an output")
endif()

set(counts "${OUTPUT_DIR}/counts.json")
file(WRITE "${counts}" "{\"schema_version\":1,\"counts\":{\"00\":2,\"10\":1,\"11\":1}}\n")
execute_process(
  COMMAND "${CLI}" evaluate --manifest "${manifest_path}" --counts -
  INPUT_FILE "${counts}"
  RESULT_VARIABLE evaluation_result
  OUTPUT_VARIABLE evaluation_output
  ERROR_VARIABLE evaluation_error)
if(NOT evaluation_result EQUAL 0)
  message(
    FATAL_ERROR
      "evaluation failed with exit code ${evaluation_result}:\n${evaluation_output}${evaluation_error}"
  )
endif()
string(JSON evaluated_case_id GET "${evaluation_output}" case_id)
string(JSON total_variation_distance GET "${evaluation_output}" metrics total_variation_distance)
if(NOT evaluated_case_id STREQUAL case_id OR total_variation_distance GREATER 1e-15)
  message(FATAL_ERROR "evaluation did not use the generated manifest reference")
endif()

execute_process(
  COMMAND "${CLI}" generate --instance-specification - --format jeff --output "${qc_directory}"
  INPUT_FILE "${instance_specification}"
  RESULT_VARIABLE jeff_result
  OUTPUT_VARIABLE jeff_output
  ERROR_VARIABLE jeff_error)
if(NOT jeff_result EQUAL 0)
  message(
    FATAL_ERROR "jeff generation failed with exit code ${jeff_result}:\n${jeff_output}${jeff_error}"
  )
endif()
string(JSON jeff_case_id GET "${jeff_output}" case_id)
string(JSON jeff_path GET "${jeff_output}" program_path)
string(JSON jeff_manifest_path GET "${jeff_output}" manifest_path)
if(NOT jeff_case_id STREQUAL case_id
   OR NOT EXISTS "${jeff_path}"
   OR NOT EXISTS "${jeff_manifest_path}")
  message(FATAL_ERROR "jeff generation changed the semantic case or omitted its output")
endif()
if(jeff_manifest_path STREQUAL manifest_path)
  message(FATAL_ERROR "QC and jeff generation reused one manifest path")
endif()
file(SIZE "${jeff_path}" jeff_size)
if(jeff_size EQUAL 0)
  message(FATAL_ERROR "jeff generation produced an empty file")
endif()

set(invalid_format_directory "${OUTPUT_DIR}/invalid-format")
run_failure(
  "invalid output format"
  "${CLI}"
  generate
  --instance-specification
  "${instance_specification}"
  --format
  invalid
  --output
  "${invalid_format_directory}")
if(EXISTS "${invalid_format_directory}")
  message(FATAL_ERROR "an invalid output format created its output directory")
endif()

set(invalid_instance_specification "${OUTPUT_DIR}/invalid-instance-specification.json")
set(invalid_directory "${OUTPUT_DIR}/invalid")
file(WRITE "${invalid_instance_specification}"
     "{\"schema_version\":1,\"benchmark\":\"unknown\",\"parameters\":{}}\n")
run_failure(
  "invalid instance specification"
  "${CLI}"
  generate
  --instance-specification
  "${invalid_instance_specification}"
  --format
  qc
  --output
  "${invalid_directory}")
file(GLOB invalid_outputs "${invalid_directory}/*")
if(invalid_outputs)
  message(FATAL_ERROR "an invalid instance specification left a final output")
endif()
