# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copy dependencies shipped beside a shared library, leaving system libraries in place.
file(REAL_PATH "${LIBRARY}" library)
cmake_path(GET library PARENT_PATH library_directory)
file(GET_RUNTIME_DEPENDENCIES LIBRARIES "${library}" RESOLVED_DEPENDENCIES_VAR dependencies)
foreach(dependency IN LISTS dependencies)
  cmake_path(GET dependency PARENT_PATH directory)
  if(directory STREQUAL library_directory)
    file(
      COPY "${dependency}"
      DESTINATION "${DESTINATION}"
      FOLLOW_SYMLINK_CHAIN)
  endif()
endforeach()
