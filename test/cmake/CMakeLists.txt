# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

set(MQT_CORE_EMBEDDED_CONFIGURE_ARGS
    -DMQT_CORE_SOURCE_DIR=${PROJECT_SOURCE_DIR} -DMLIR_DIR=${MLIR_DIR}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})

# Reuse dependency sources populated by the outer build so this configure-only regression test does
# not require network access.
foreach(
  dependency
  jeff-mlir
  jeff
  capnproto
  nlohmann_json
  boost_mp
  qdmi
  spdlog)
  set(dependency_source_dir "${FETCHCONTENT_BASE_DIR}/${dependency}-src")
  if(EXISTS "${dependency_source_dir}")
    string(TOUPPER "${dependency}" dependency_name)
    list(APPEND MQT_CORE_EMBEDDED_CONFIGURE_ARGS
         "-DFETCHCONTENT_SOURCE_DIR_${dependency_name}=${dependency_source_dir}")
  endif()
endforeach()

foreach(build_devices OFF ON)
  string(TOLOWER "${build_devices}" build_devices_lower)
  set(embedded_configure_command
      ${CMAKE_COMMAND} --fresh -S ${CMAKE_CURRENT_SOURCE_DIR}/embedded -B
      ${CMAKE_CURRENT_BINARY_DIR}/embedded-devices-${build_devices_lower} -G "${CMAKE_GENERATOR}"
      ${MQT_CORE_EMBEDDED_CONFIGURE_ARGS} -DEMBEDDED_BUILD_MQT_CORE_QDMI_DEVICES=${build_devices})
  if(CMAKE_GENERATOR_PLATFORM)
    list(APPEND embedded_configure_command -A "${CMAKE_GENERATOR_PLATFORM}")
  endif()
  if(CMAKE_GENERATOR_TOOLSET)
    list(APPEND embedded_configure_command -T "${CMAKE_GENERATOR_TOOLSET}")
  endif()
  add_test(NAME mqt-core-embedded-configure-devices-${build_devices_lower}
           COMMAND ${embedded_configure_command})
endforeach()
