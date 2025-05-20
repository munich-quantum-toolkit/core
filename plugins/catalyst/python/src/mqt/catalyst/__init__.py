# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Catalyst Plugin."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pennylane as qml

from catalyst.passes import PassPlugin

from ._version import version as __version__
from ._version import version_tuple as version_info


def get_catalyst_plugin_abs_path() -> Path:
    try:
        dist = distribution("mqt-core")
        # Check for the plugin in the mqt-core package on Linux
        catalyst_plugin_abs_path = Path(dist.locate_file("mqt/core/lib/mqt-catalyst-plugin.so"))
        if catalyst_plugin_abs_path.exists() and catalyst_plugin_abs_path.is_file():
            return catalyst_plugin_abs_path
        # Check for the plugin in the mqt-core package on macOS
        catalyst_plugin_abs_path = Path(dist.locate_file("mqt/core/lib/mqt-catalyst-plugin.dylib"))
        if catalyst_plugin_abs_path.exists() and catalyst_plugin_abs_path.is_file():
            return catalyst_plugin_abs_path
        msg = "mqt-catalyst-plugin library not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "mqt-catalyst-plugin not installed, installation required to access the MQT Catalyst Plugin library."
        raise ImportError(msg) from None


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
