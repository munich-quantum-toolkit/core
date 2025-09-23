# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(NOT DEFINED DOCS_DIR)
  message(FATAL_ERROR "DOCS_DIR is not defined!")
endif()

file(GLOB_RECURSE MD_FILES "${DOCS_DIR}/*.md")
foreach(MD_FILE ${MD_FILES})
  # Read the entire file content into a single variable.
  file(READ ${MD_FILE} CONTENT)

  # Replace lines that only contain [TOC], allowing for whitespace.
  string(REGEX REPLACE "\n\\[TOC\\]\n" "" CONTENT "${CONTENT}")

  # Replace lines that only contain an llvm-project source link, allowing for whitespace.
  string(REGEX REPLACE "\n\\[source\\]\\(https://github.com/llvm/llvm-project/blob/main.*\.td\\)\n"
                       "" CONTENT "${CONTENT}")

  # Write the processed content back to the file.
  file(WRITE ${MD_FILE} "${CONTENT}")
endforeach()
