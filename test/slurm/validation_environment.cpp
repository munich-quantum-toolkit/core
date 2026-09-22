/*
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <https://www.gnu.org/licenses/>.
 */

extern "C" {
#include <slurm/slurm_version.h>
#include <slurm/spank.h>
}

// This fixture creates different provider inputs before the production hook.
// Names and signatures are fixed by Slurm's plugin ABI.
// NOLINTBEGIN(readability-identifier-naming, misc-use-internal-linkage,
// cppcoreguidelines-avoid-c-arrays)
extern "C" {
extern const char plugin_name[] = "mqt_test_validation_environment";
extern const char plugin_type[] = "spank";
extern const unsigned int plugin_version = SLURM_VERSION_NUMBER;
extern const unsigned int spank_plugin_version = 1;

int slurm_spank_task_init(spank_t spank, int /*count*/, char* /*arguments*/[]) {
  int task = 0;
  // Slurm exposes item lookup through a variadic C ABI.
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
  if (spank_get_item(spank, S_TASK_ID, &task) != ESPANK_SUCCESS) {
    return -1;
  }
  if (task == 1 && spank_setenv(spank, "MQT_SLURM_TEST_REFERENCE",
                                "task-specific", 1) != ESPANK_SUCCESS) {
    return -1;
  }
  return ESPANK_SUCCESS;
}
}
// NOLINTEND(readability-identifier-naming, misc-use-internal-linkage,
// cppcoreguidelines-avoid-c-arrays)
