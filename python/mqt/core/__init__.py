# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core - The Backbone of the Munich Quantum Toolkit."""

from __future__ import annotations

__lazy_modules__ = {"mqt.core._version", "mqt.core._warnings"}

from mqt.core._version import version as __version__
from mqt.core._version import version_tuple as version_info
from mqt.core._warnings import UntestedVersionWarning

__all__ = ["UntestedVersionWarning", "__version__", "version_info"]
