# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Check that a generated QDMI registry file contains every expected fragment. MANIFEST names the
# file. EXPECTED holds the fragments, separated by '|' so that a fragment may contain ';'.

if(NOT MANIFEST OR NOT EXPECTED)
  message(FATAL_ERROR "verify_generated_manifest.cmake requires MANIFEST and EXPECTED")
endif()
if(NOT EXISTS "${MANIFEST}")
  message(FATAL_ERROR "Missing generated QDMI registry file: ${MANIFEST}")
endif()

file(READ "${MANIFEST}" manifest_content)
string(REPLACE "|" ";" expected_fragments "${EXPECTED}")
foreach(fragment IN LISTS expected_fragments)
  string(FIND "${manifest_content}" "${fragment}" fragment_index)
  if(fragment_index EQUAL -1)
    message(FATAL_ERROR "${MANIFEST} does not contain '${fragment}':\n${manifest_content}")
  endif()
endforeach()
