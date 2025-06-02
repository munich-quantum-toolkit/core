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
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pennylane as qml

from catalyst.passes import PassPlugin

from ._version import version as __version__
from ._version import version_tuple as version_info


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

    # 1. Try site-packages from distribution metadata (wheel install)
    try:
        dist = distribution("mqt-catalyst-plugin")
        lib_base = Path(dist.locate_file("mqt/catalyst/lib"))
        for path in lib_base.glob(f"**/mqt-catalyst-plugin{ext}"):
            if path.is_file():
                return path.resolve()
    except PackageNotFoundError:
        pass

    # 2. Fallback: search in editable build directory under the repo root
    current = Path(__file__).resolve()
    for up in [2, 3, 4]:  # src/, src/mqt/, plugins/catalyst/python/src etc.
        candidate_root = current.parents[up] / "build"
        if candidate_root.exists():
            matches = list(candidate_root.glob(f"**/plugins/catalyst/lib/mqt-catalyst-plugin{ext}"))
            if matches:
                return matches[0].resolve()

    # 3. Give up
    msg = f"mqt-catalyst-plugin{ext} not found in site-packages or build directories."
    raise FileNotFoundError(msg)


def name2pass(_name: str) -> tuple[Path, str]:
    """Convert a pass name to its plugin path and pass name (required by Catalyst).

    Args:
        _name: The name of the pass, e.g., "mqt-core-round-trip".

    Returns:
        A tuple containing the absolute path to the plugin and the pass name.
    """
    return get_catalyst_plugin_abs_path(), "mqt-core-round-trip"


def mqt_core_roundtrip(*flags: any, **valued_options: any) -> qml.QNode:
    """Decorator to apply the MQT Core Round Trip pass to a QNode.

    This pass ensures that the QNode can be processed in a round-trip
    fashion with the MQT Catalyst plugin.

    Args:
        *flags: Optional flags to pass to the pass.
        **valued_options: Optional keyword arguments to pass to the pass.

    Returns:
        A decorator that applies the MQT Core Round Trip pass to a QNode.
    """

    def add_pass_to_pipeline(**kwargs: any) -> list[PassPlugin]:
        pass_pipeline = kwargs.get("pass_pipeline", [])
        pass_pipeline.append(
            PassPlugin(
                get_catalyst_plugin_abs_path(),
                "mqt-core-round-trip",
                *flags,
                **valued_options,
            )
        )
        return pass_pipeline

    def decorator(qnode: qml.QNode) -> qml.QNode:
        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            msg = f"A QNode is expected, got the classical function {qnode}"
            raise TypeError(msg)

        def qnode_call(*args: any, **kwargs: any) -> qml.QNode:
            kwargs["pass_pipeline"] = add_pass_to_pipeline(**kwargs)
            return qnode(*args, **kwargs)

        return qnode_call

    # When the decorator is used without ()
    if len(flags) == 1 and isinstance(flags[0], qml.QNode):
        qnode = flags[0]

        def qnode_call(*args: any, **kwargs: any) -> qml.QNode:
            kwargs["pass_pipeline"] = add_pass_to_pipeline(**kwargs)
            return qnode(*args, **kwargs)

        return qnode_call

    # When the decorator is used with ()
    return decorator


__all__ = ["__version__", "get_catalyst_plugin_abs_path", "mqt_core_roundtrip", "name2pass", "version_info"]
