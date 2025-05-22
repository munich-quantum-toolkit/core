# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

function(kebab_to_camel output input)
  string(REPLACE "-" ";" parts "${input}")
  set(result "")
  foreach(part ${parts})
    string(SUBSTRING ${part} 0 1 first)
    string(SUBSTRING ${part} 1 -1 rest)
    string(TOUPPER ${first} first)
    string(APPEND result "${first}${rest}")
  endforeach()
  set(${output}
      "${result}"
      PARENT_SCOPE)
endfunction()

function(add_mqt_core_library name)
  cmake_parse_arguments(ARG "" "ALIAS_NAME" "" ${ARGN})
  add_library(${name} $<$<BOOL:${BUILD_MQT_CORE_SHARED_LIBS}>:SHARED> ${ARG_UNPARSED_ARGUMENTS})
  if(NOT ARG_ALIAS_NAME)
    # remove prefix 'mqt-' from target name if exists
    string(REGEX REPLACE "^${MQT_CORE_TARGET_NAME}" "" ALIAS_NAME_ARG ${name})
    # transform kebab-case to camelCase
    kebab_to_camel(ARG_ALIAS_NAME ${ALIAS_NAME_ARG})
  endif()
  add_library(MQT::Core${ARG_ALIAS_NAME} ALIAS ${name})
endfunction()
