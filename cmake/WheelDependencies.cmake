# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# cibuildwheel clears this marker before each wheel series. Once CMake has
# generated the first build, every ABI builds against the same pinned sources.
set(_wheel_deps_ready "${CMAKE_BINARY_DIR}/dependencies-ready")
if(EXISTS "${_wheel_deps_ready}")
  set(FETCHCONTENT_FULLY_DISCONNECTED ON)
endif()
file(GENERATE OUTPUT "${_wheel_deps_ready}" CONTENT "")
