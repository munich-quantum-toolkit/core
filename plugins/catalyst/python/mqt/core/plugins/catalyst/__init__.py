# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Catalyst Plugin."""

from __future__ import annotations

import platform
from importlib import resources
from pathlib import Path


def get_catalyst_plugin_abs_path() -> Path:
    """Locate the mqt-catalyst-plugin shared library.

    Returns:
        The absolute path to the plugin shared library.

    Raises:
        FileNotFoundError: If the plugin library is not found.
    """
    ext = {"Darwin": ".dylib", "Linux": ".so", "Windows": ".dll"}.get(platform.system(), ".so")

    # 0. Allow override via env variable
    from os import getenv

    override = getenv("MQT_CATALYST_PLUGIN_PATH")
    if override:
        override_path = Path(override)
        if override_path.exists():
            return override_path.resolve()
        msg = f"Environment override MQT_CATALYST_PLUGIN_PATH is set but not valid: {override_path}"
        raise FileNotFoundError(msg)

    # First check in the package resources
    for file in resources.files("mqt.core.plugins.catalyst").iterdir():
        if "mqt-core-catalyst-plugin" in file.name:
            return Path(str(file)).resolve()

    # Then check in site-packages
    import site

    for site_dir in site.getsitepackages():
        site_path = Path(site_dir) / "mqt/core/plugins/catalyst"
        if site_path.exists():
            for file in site_path.iterdir():
                if "mqt-core-catalyst-plugin" in file.name:
                    return file.resolve()

    msg = f"Could not locate catalyst plugin library with extension {ext}"
    raise FileNotFoundError(msg)


def name2pass(name: str) -> tuple[Path, str]:
    """Convert a pass name to its plugin path and pass name (required by Catalyst).

    Args:
        name: The name of the pass, e.g., "mqt-core-round-trip".

    Returns:
        A tuple containing the absolute path to the plugin and the pass name.
    """
    return get_catalyst_plugin_abs_path(), name


__all__ = ["get_catalyst_plugin_abs_path", "name2pass"]
