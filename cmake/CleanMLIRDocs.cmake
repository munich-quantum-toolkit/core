# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
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
  file(READ ${MD_FILE} CONTENT)

  # Sphinx supplies navigation for the generated reference pages.
  string(REGEX REPLACE "\n\\[TOC\\]\n" "" CONTENT "${CONTENT}")

  # TableGen links local definitions to llvm-project, where those files do not exist.
  string(REGEX REPLACE "\n\\[source\\]\\(https://github.com/llvm/llvm-project/blob/main.*\.td\\)\n"
                       "" CONTENT "${CONTENT}")

  file(WRITE ${MD_FILE} "${CONTENT}")
endforeach()
