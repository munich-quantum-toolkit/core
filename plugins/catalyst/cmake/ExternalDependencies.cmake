# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Declare all external dependencies and make sure that they are available.

set(CATALYST_VERSION 0.12.0)
# Check if the pennylane-catalyst package is installed in the python environment.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import catalyst; print(catalyst.__version__)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE FOUND_CATALYST_VERSION)
if(FOUND_CATALYST_VERSION)
  message(STATUS "Found pennylane-catalyst ${CATALYST_VERSION} in python environment.")
  # Check if the version is compatible.
  if(FOUND_CATALYST_VERSION VERSION_LESS ${CATALYST_VERSION})
    message(
      WARNING
        "pennylane-catalyst version ${FOUND_CATALYST_VERSION} in python environment is not compatible."
    )
  else()
    # Detect the installed catalyst include files.
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c
              "import catalyst.utils.runtime_environment as c; print(c.get_include_path())"
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE CATALYST_INCLUDE_DIRS)
  endif()
else()
  # Unfortunately, the download for an individual package cannot be turned off. To avoid downloading
  # the entire package, we use `find_package` instead.
  find_package(Catalyst ${CATALYST_VERSION} REQUIRED)
endif()

if(NOT CATALYST_INCLUDE_DIRS)
  message(
    FATAL_ERROR
      "The include directory of the pennylane-catalyst package could not be retrieved. Please ensure that the catalyst is installed correctly."
  )
endif()
