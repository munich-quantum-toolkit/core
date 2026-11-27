# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Version of the installed MQT Core package."""

from importlib import metadata

__version__ = version = metadata.version("mqt-core")
__version_tuple__ = version_tuple = tuple(int(part) if part.isdecimal() else part for part in version.split("."))
