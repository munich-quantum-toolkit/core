# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# A cmake script to determine the project version either
#
# * from the `SKBUILD_PROJECT_VERSION` variable (if available) or
# * from the git repository (if available) or
# * from the Python package version (if available).
#
# The following variables are set:
#
# * MQT_CORE_VERSION
# * MQT_CORE_VERSION_STRING
# * MQT_CORE_VERSION_FOUND (TRUE if version was found, FALSE otherwise)

function(version_from_skbuild)
  if(NOT MQT_CORE_MASTER_PROJECT)
    message(VERBOSE
            "Not the master project. Cannot determine project version from scikit-build-core.")
    return()
  endif()

  if(NOT DEFINED SKBUILD_PROJECT_VERSION)
    message(
      VERBOSE
      "SKBUILD_PROJECT_VERSION not found. Cannot determine project version from scikit-build-core.")
    return()
  endif()
  set(MQT_CORE_VERSION_STRING
      "${SKBUILD_PROJECT_VERSION}"
      CACHE INTERNAL "MQT_CORE_VERSION_STRING")

  # Regex for package version: Major.Minor.Patch(.dev<commits>+g<sha1>.d<dirty>)
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)(\\.dev([0-9]+)\\+g([0-9a-f]+)(\\.d[0-9]+)?)?"
               MQT_CORE_VERSION_STRING "${MQT_CORE_VERSION_STRING}")
  set(MQT_CORE_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(MQT_CORE_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(MQT_CORE_VERSION_PATCH "${CMAKE_MATCH_3}")

  if("${MQT_CORE_VERSION_MAJOR}" STREQUAL ""
     OR "${MQT_CORE_VERSION_MINOR}" STREQUAL ""
     OR "${MQT_CORE_VERSION_PATCH}" STREQUAL "")
    message(
      VERBOSE
      "SKBUILD_PROJECT_VERSION invalid. Cannot determine project version from scikit-build-core.")
    return()
  endif()

  set(MQT_CORE_VERSION_FOUND
      TRUE
      CACHE INTERNAL "MQT_CORE_VERSION_FOUND")
  set(MQT_CORE_VERSION
      "${MQT_CORE_VERSION_MAJOR}.${MQT_CORE_VERSION_MINOR}.${MQT_CORE_VERSION_PATCH}"
      CACHE INTERNAL "MQT_CORE_VERSION")
  return()

endfunction()

function(version_from_git)
  find_package(Git QUIET)
  if(NOT GIT_FOUND)
    message(STATUS "Git not found. Cannot determine project version from Git.")
    return()
  endif()

  set(GIT_DIR "${PROJECT_SOURCE_DIR}/.git")
  message(VERBOSE "Looking for Git repository at ${GIT_DIR}")
  if(NOT EXISTS "${GIT_DIR}")
    message(VERBOSE "Git repository not found. Cannot determine project version from Git.")
    return()
  endif()

  # Get the latest abbreviated commit hash of the working branch
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE MQT_CORE_VERSION_SHA1
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT "${MQT_CORE_VERSION_SHA1}" STREQUAL "")
    message(VERBOSE "Git commit hash: ${MQT_CORE_VERSION_SHA1}")
  else()
    message(VERBOSE "Git commit hash not found. Cannot determine project version from Git.")
    return()
  endif()

  # Get the latest tag name
  execute_process(
    COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE MQT_CORE_VERSION_TAG
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT "${MQT_CORE_VERSION_TAG}" STREQUAL "")
    message(VERBOSE "Git tag: ${MQT_CORE_VERSION_TAG}")
  else()
    message(VERBOSE "Git tag not found. Cannot determine project version from Git.")
    return()
  endif()

  # Get the number of commits since the latest tag
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-list ${MQT_CORE_VERSION_TAG}..HEAD --count
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE MQT_CORE_VERSION_COMMITS
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT "${MQT_CORE_VERSION_COMMITS}" STREQUAL "")
    message(VERBOSE "Git commits since tag: ${MQT_CORE_VERSION_COMMITS}")
  else()
    message(VERBOSE "Git commits since tag not found. Cannot determine project version from Git.")
    return()
  endif()

  # Parse the version number from the latest tag (v.MAJOR.MINOR.PATCH)
  string(REGEX MATCH "v([0-9]+)\\.([0-9]+)\\.([0-9]+)" MQT_CORE_VERSION_TAG
               "${MQT_CORE_VERSION_TAG}")
  set(MQT_CORE_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(MQT_CORE_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(MQT_CORE_VERSION_PATCH "${CMAKE_MATCH_3}")

  if(${MQT_CORE_VERSION_MAJOR} STREQUAL ""
     OR ${MQT_CORE_VERSION_MINOR} STREQUAL ""
     OR ${MQT_CORE_VERSION_PATCH} STREQUAL "")
    message(VERBOSE
            "Git version information incomplete. Cannot determine project version from Git.")
    return()
  endif()

  # if commits since tag, increment patch version
  if(NOT "${MQT_CORE_VERSION_COMMITS}" STREQUAL "0")
    math(EXPR MQT_CORE_VERSION_PATCH "${MQT_CORE_VERSION_PATCH}+1")
  endif()
  set(MQT_CORE_VERSION
      "${MQT_CORE_VERSION_MAJOR}.${MQT_CORE_VERSION_MINOR}.${MQT_CORE_VERSION_PATCH}"
      CACHE INTERNAL "MQT_CORE_VERSION")
  set(MQT_CORE_VERSION_STRING "${MQT_CORE_VERSION}")
  # if commits since tag, append version info
  if(NOT "${MQT_CORE_VERSION_COMMITS}" STREQUAL "0")
    set(MQT_CORE_VERSION_STRING
        "${MQT_CORE_VERSION}.dev${MQT_CORE_VERSION_COMMITS}+g${MQT_CORE_VERSION_SHA1}"
        CACHE INTERNAL "MQT_CORE_VERSION_STRING")
  endif()

  set(MQT_CORE_VERSION_FOUND
      TRUE
      CACHE INTERNAL "MQT_CORE_VERSION_FOUND")
endfunction()

function(version_from_package)
  if(NOT TARGET Python::Interpreter)
    message(VERBOSE
            "Python interpreter not found. Cannot determine project version from Python package.")
    return()
  endif()

  execute_process(
    COMMAND ${Python_EXECUTABLE} -m mqt.core --version
    OUTPUT_VARIABLE MQT_CORE_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)

  if("${MQT_CORE_VERSION_STRING}" STREQUAL "")
    message(
      VERBOSE
      "Python package version not found. Cannot determine project version from Python package.")
    return()
  endif()
  set(MQT_CORE_VERSION_STRING
      "${MQT_CORE_VERSION_STRING}"
      CACHE INTERNAL "MQT_CORE_VERSION_STRING")

  # Regex for package version: Major.Minor.Patch(.dev<commits>+g<sha1>.d<dirty>)
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)(\\.dev([0-9]+)\\+g([0-9a-f]+)(\\.d[0-9]+)?)?"
               MQT_CORE_VERSION_STRING "${MQT_CORE_VERSION_STRING}")
  set(MQT_CORE_VERSION_MAJOR "${CMAKE_MATCH_1}")
  set(MQT_CORE_VERSION_MINOR "${CMAKE_MATCH_2}")
  set(MQT_CORE_VERSION_PATCH "${CMAKE_MATCH_3}")

  if(NOT MQT_CORE_VERSION_MAJOR
     OR NOT MQT_CORE_VERSION_MINOR
     OR NOT MQT_CORE_VERSION_PATCH)
    message(
      VERBOSE
      "Python package version information incomplete. Cannot determine project version from Python package."
    )
    return()
  endif()

  set(MQT_CORE_VERSION
      "${MQT_CORE_VERSION_MAJOR}.${MQT_CORE_VERSION_MINOR}.${MQT_CORE_VERSION_PATCH}"
      CACHE INTERNAL "MQT_CORE_VERSION")
  set(MQT_CORE_VERSION_FOUND
      TRUE
      CACHE INTERNAL "MQT_CORE_VERSION_FOUND")
endfunction()

function(get_mqt_core_version)
  # Initialize as not found
  set(MQT_CORE_VERSION_FOUND
      FALSE
      CACHE INTERNAL "MQT_CORE_VERSION_FOUND")

  if(MQT_CORE_MASTER_PROJECT)
    version_from_skbuild()
    if(MQT_CORE_VERSION_FOUND)
      message(
        STATUS
          "Found project version ${MQT_CORE_VERSION} from scikit-build-core (full version: ${MQT_CORE_VERSION_STRING})"
      )
      return()
    endif()
  endif()

  version_from_git()
  if(MQT_CORE_VERSION_FOUND)
    message(
      STATUS
        "Found project version ${MQT_CORE_VERSION} from git (full version: ${MQT_CORE_VERSION_STRING})"
    )
    return()
  endif()

  version_from_package()
  if(MQT_CORE_VERSION_FOUND)
    message(
      STATUS
        "Found project version ${MQT_CORE_VERSION} from Python package (full version: ${MQT_CORE_VERSION_STRING})"
    )
    return()
  endif()

  message(WARNING "Could not determine project version. Setting to 0.0.0")
  set(MQT_CORE_VERSION
      "0.0.0"
      CACHE INTERNAL "MQT_CORE_VERSION")
  set(MQT_CORE_VERSION_STRING
      "0.0.0"
      CACHE INTERNAL "MQT_CORE_VERSION_STRING")
endfunction()
