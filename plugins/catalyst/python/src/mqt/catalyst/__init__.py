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
    return get_catalyst_plugin_abs_path(), "mqt-core-round-trip"


def MQTCoreRoundTrip(*flags, **valued_options):
    def add_pass_to_pipeline(**kwargs):
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

    def decorator(qnode):
        if not isinstance(qnode, qml.QNode):
            # Technically, this apply pass is general enough that it can apply to
            # classical functions too. However, since we lack the current infrastructure
            # to denote a function, let's limit it to qnodes
            msg = f"A QNode is expected, got the classical function {qnode}"
            raise TypeError(msg)

        def qnode_call(*args, **kwargs):
            kwargs["pass_pipeline"] = add_pass_to_pipeline(**kwargs)
            return qnode(*args, **kwargs)

        return qnode_call

    # When the decorator is used without ()
    if len(flags) == 1 and isinstance(flags[0], qml.QNode):
        qnode = flags[0]

        def qnode_call(*args, **kwargs):
            kwargs["pass_pipeline"] = add_pass_to_pipeline(**kwargs)
            return qnode(*args, **kwargs)

        return qnode_call

    # When the decorator is used with ()
    return decorator


__all__ = ["MQTCoreRoundTrip", "__version__", "get_catalyst_plugin_abs_path", "name2pass", "version_info"]
