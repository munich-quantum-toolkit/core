# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

if(MQT_CORE_INSTALL)
  if(APPLE)
    set(BASEPOINT @loader_path)
  else()
    set(BASEPOINT ${ORIGIN})
  endif()
  set(CMAKE_INSTALL_RPATH ${BASEPOINT} ${BASEPOINT}/${CMAKE_INSTALL_LIBDIR})
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# add the IR bindings package
add_subdirectory(ir)

# add the DD bindings package
add_subdirectory(dd)
