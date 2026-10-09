# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

get_filename_component(device_dir "${device}" DIRECTORY)
get_filename_component(device_name "${device}" NAME)
set(asset "${build_dir}/inputs/metadata-runtime.json")

# Neither an asset edit nor an updated device should require relinking the consumer.
file(APPEND "${asset}" "\n")
file(APPEND "${device}" "\n")
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${build_dir}" --config "${config}" --target
                        runtime-consumer COMMAND_ERROR_IS_FATAL ANY)
foreach(staged_asset IN ITEMS "${device_dir}/metadata-runtime.json"
                              "${consumer_dir}/metadata-runtime.json")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${asset}" "${staged_asset}"
                          COMMAND_ERROR_IS_FATAL ANY)
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${device}"
                        "${consumer_dir}/${device_name}" COMMAND_ERROR_IS_FATAL ANY)
