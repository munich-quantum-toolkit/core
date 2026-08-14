# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optional dependency checks."""

from __future__ import annotations

from importlib import import_module

__all__ = ["is_module_available"]


def is_module_available(module_name: str) -> bool:
    """Return whether a top-level module can be imported.

    Raises:
        ModuleNotFoundError: If an import required by the requested module is missing.
    """
    try:
        import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return False
        raise
    return True
