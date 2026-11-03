# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Public qdmi module backed by the shared native extension."""

import sys

from mqt.core import _core
from mqt.core._qdmi_discovery import discover_qdmi_manifests

qdmi = _core.qdmi
qdmi.__file__ = _core.__file__
sys.modules[__name__] = qdmi

discover_qdmi_manifests(qdmi.builtin_driver.add_manifest)
