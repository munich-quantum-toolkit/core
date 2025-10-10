# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

add_subdirectory(fomac)

# install device.json
install(
  FILES ${PROJECT_SOURCE_DIR}/json/na/device.json
  DESTINATION ./na
  COMPONENT ${MQT_CORE_TARGET_NAME}_Python)

# install the Python stub files in editable mode for better IDE support
if(SKBUILD_STATE STREQUAL "editable")
  file(GLOB_RECURSE PYI_FILES ${PROJECT_SOURCE_DIR}/python/mqt/core/na/*.pyi)
  install(
    FILES ${PYI_FILES}
    DESTINATION ./na
    COMPONENT ${MQT_CORE_TARGET_NAME}_Python)
endif()
