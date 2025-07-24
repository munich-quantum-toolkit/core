# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Declare all external dependencies and make sure that they are available.

include(FetchContent)

if(DEFINED Python_EXECUTABLE AND Python_EXECUTABLE)
  message(STATUS "Using Python executable: ${Python_EXECUTABLE}")
  set(CATALYST_VERSION 0.12.0)
  # Check if the pennylane-catalyst package is installed in the python environment.
  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import site; import sys; sys.path.extend(site.getsitepackages()); import catalyst; print(catalyst.__version__)"
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

      message(STATUS "Catalyst include path resolved to: ${CATALYST_INCLUDE_DIRS}")

      string(FIND "${CATALYST_INCLUDE_DIRS}" "site-packages" SITEPKG_IDX)

      if(SITEPKG_IDX EQUAL -1)
        # In case of a installation from source Assume include path looks like: <root>/mlir/include
        # Derive <root>/mlir/build/include and <root>/mlir/build/lib/cmake/catalyst
        get_filename_component(CATALYST_MLIR_ROOT "${CATALYST_INCLUDE_DIRS}/.." ABSOLUTE)
        set(CATALYST_BUILD_DIR "${CATALYST_MLIR_ROOT}/build")
        set(CATALYST_BUILD_INCLUDE_DIR "${CATALYST_BUILD_DIR}/include")
        set(Catalyst_DIR "${CATALYST_BUILD_DIR}/lib/cmake/catalyst")

        include_directories("${CATALYST_INCLUDE_DIRS}")
        include_directories("${CATALYST_BUILD_INCLUDE_DIR}")
      endif()

    endif()
  else()
    # Unfortunately, the download for an individual package cannot be turned off. To avoid
    # downloading the entire package, we use `find_package` instead.
    find_package(Catalyst ${CATALYST_VERSION} REQUIRED)
  endif()

  if(NOT CATALYST_INCLUDE_DIRS)
    message(
      FATAL_ERROR
        "The include directory of the pennylane-catalyst package could not be retrieved. Please ensure that the catalyst is installed correctly."
    )
  endif()

  # execute_process( COMMAND "${Python_EXECUTABLE}" -c "import mqt.qmap;
  # print(mqt.qmap.__version__)" OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE FOUND_QMAP_VERSION
  # ERROR_QUIET) if(FOUND_QMAP_VERSION) message(STATUS "Found mqt.qmap ${FOUND_QMAP_VERSION} in
  # python environment.") if(FOUND_QMAP_VERSION VERSION_LESS ${MQT_QMAP_VERSION}) message( WARNING
  # "mqt.qmap version ${FOUND_QMAP_VERSION} in python environment is not compatible. Expected >=
  # ${MQT_QMAP_VERSION}." ) endif() else() message( WARNING "mqt.qmap not found in python
  # environment. Please install it via pip install mqt.qmap" ) endif()

else()
  message(
    FATAL_ERROR
      "Python executable is not defined. Please set the Python_EXECUTABLE variable to the path of the Python interpreter."
  )
endif()

# cmake-format: off
set(MQT_CORE_MINIMUM_VERSION 3.1.0
    CACHE STRING "MQT Core minimum version")
set(MQT_CORE_VERSION 3.1.1
    CACHE STRING "MQT Core version")
set(MQT_CORE_REV "4313e1ab88b6598e78c062054c70d827f358b690"
    CACHE STRING "MQT Core identifier (tag, branch or commit hash)")
set(MQT_CORE_REPO_OWNER "munich-quantum-toolkit"
    CACHE STRING "MQT Core repository owner (change when using a fork)")
# cmake-format: on
set(BUILD_MQT_CORE_TESTS
    OFF
    CACHE BOOL "Build MQT Core tests")
set(BUILD_MQT_CORE_SHARED_LIBS
    OFF
    CACHE BOOL "Build MQT Core shared libraries")
set(BUILD_MQT_CORE_MLIR
    ON
    CACHE BOOL "Build MQT Core MLIR support")
set(CMAKE_POSITION_INDEPENDENT_CODE
    ON
    CACHE BOOL "Enable position independent code (PIC) for MQT Core")

message(STATUS "[mqt-core] Declaring MQT Core dependency...")
message(STATUS "[mqt-core] Requested version: ${MQT_CORE_VERSION}")
message(STATUS "[mqt-core] Git tag/rev: ${MQT_CORE_REV}")
message(STATUS "[mqt-core] Source dir override: ${CMAKE_CURRENT_SOURCE_DIR}/../..")

# Override path (ensures reuse)
set(FETCHCONTENT_SOURCE_DIR_MQT-CORE
    "${CMAKE_CURRENT_SOURCE_DIR}/../.."
    CACHE PATH "Source directory for MQT Core")
FetchContent_Declare(
  mqt-core
  GIT_REPOSITORY https://github.com/${MQT_CORE_REPO_OWNER}/core.git
  GIT_TAG ${MQT_CORE_REV}
  FIND_PACKAGE_ARGS ${MQT_CORE_MINIMUM_VERSION})
list(APPEND FETCH_PACKAGES mqt-core)

message(STATUS "[mqt-core] Fetching and making available mqt-core...")
FetchContent_MakeAvailable(mqt-core)

# Re-expose the real source directory for downstream reuse
set(FETCHCONTENT_SOURCE_DIR_MQT-CORE "${mqt-core_SOURCE_DIR}")
message(STATUS "[mqt-core] Final source directory: ${mqt-core_SOURCE_DIR}")

# cmake-format: off
set(MQT_QMAP_VERSION 3.2.0
    CACHE STRING "MQT QMAP version")
set(MQT_QMAP_REV "v3.2.0"
    CACHE STRING "MQT QMAP identifier (tag, branch or commit hash)")
set(MQT_QMAP_REPO_OWNER "cda-tum"
    CACHE STRING "MQT QMAP repository owner (change when using a fork)")
# cmake-format: on

message(STATUS "[mqt-qmap] Declaring MQT QMAP dependency...")
message(STATUS "[mqt-qmap] Version: ${MQT_QMAP_VERSION}")
message(STATUS "[mqt-qmap] Git tag/rev: ${MQT_QMAP_REV}")

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  message(STATUS "[mqt-qmap] Using FetchContent with FIND_PACKAGE_ARGS (CMake >= 3.24)")
  FetchContent_Declare(
    mqt-qmap
    GIT_REPOSITORY https://github.com/${MQT_QMAP_REPO_OWNER}/mqt-qmap.git
    GIT_TAG ${MQT_QMAP_REV}
    FIND_PACKAGE_ARGS ${MQT_QMAP_VERSION})
  list(APPEND FETCH_PACKAGES mqt-qmap)
else()
  message(STATUS "[mqt-qmap] Trying find_package for mqt-qmap (CMake < 3.24)")
  find_package(mqt-qmap ${MQT_QMAP_VERSION} QUIET)
  if(NOT mqt-qmap_FOUND)
    message(STATUS "[mqt-qmap] Not found via find_package, falling back to FetchContent...")
    FetchContent_Declare(
      mqt-qmap
      GIT_REPOSITORY https://github.com/${MQT_QMAP_REPO_OWNER}/mqt-qmap.git
      GIT_TAG ${MQT_QMAP_REV})
    list(APPEND FETCH_PACKAGES mqt-qmap)
  else()
    message(STATUS "[mqt-qmap] Found via find_package.")
  endif()
endif()

# Final step: make all declared packages available
message(STATUS "[FetchContent] Making available: ${FETCH_PACKAGES}")
FetchContent_MakeAvailable(${FETCH_PACKAGES})

# Optional: print final resolved paths
message(STATUS "[mqt-core] Resolved path: ${mqt-core_SOURCE_DIR}")
message(STATUS "[mqt-qmap] Resolved path: ${mqt-qmap_SOURCE_DIR}")
