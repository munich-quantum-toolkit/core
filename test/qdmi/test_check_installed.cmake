# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

file(REMOVE_RECURSE "${WORK_DIR}")
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${WORK_DIR}" --config
                        "${CONFIG}" --component "${RUNTIME_COMPONENT}" COMMAND_ERROR_IS_FATAL ANY)

# Keep the build catalogue intact so loading the build-tree driver is observable.
set(manifest "${WORK_DIR}/${INSTALL_LIBDIR}/mqt-core-qdmi-sc-device.qdmi.json")
file(READ "${manifest}" catalogue)
string(REPLACE "mqt.sc.default" "installed.only" catalogue "${catalogue}")
file(WRITE "${manifest}" "${catalogue}")

foreach(device installed.only mqt.sc.default)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env --unset=MQT_CORE_QDMI_CONFIG_JSON --unset=MQT_CORE_QDMI_CONFIG_FILE
      "${WORK_DIR}/${INSTALL_BINDIR}/mqt-core-qdmi-check" --device "${device}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 10)
  if((device STREQUAL "installed.only" AND NOT result STREQUAL "0")
     OR (device STREQUAL "mqt.sc.default" AND NOT result STREQUAL "1"))
    message(
      FATAL_ERROR
        "Installed checker selected the wrong catalogue: ${device}: ${result}: ${output}${error}")
  endif()
endforeach()
