# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MQT Plugin interface."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pennylane as qml
from catalyst.passes import PassPlugin


def get_catalyst_plugin_abs_path() -> Path:
    """Returns the absolute path to the MQT plugin."""
    try:
        dist = distribution("mqt-core")
        # Check for the plugin in the mqt-core package on Linux
        catalyst_plugin_abs_path = Path(dist.locate_file("mqt/core/???/mqt-catalyst-plugin.so"))
        if catalyst_plugin_abs_path.exists() and catalyst_plugin_abs_path.is_file():
            return catalyst_plugin_abs_path
        # Check for the plugin in the mqt-core package on macOS
        catalyst_plugin_abs_path = Path(dist.locate_file("mqt/core/???/mqt-catalyst-plugin.dylib"))
        if catalyst_plugin_abs_path.exists() and catalyst_plugin_abs_path.is_file():
            return catalyst_plugin_abs_path
        # Check for the plugin in the mqt-core package on Windows
        catalyst_plugin_abs_path = Path(dist.locate_file("mqt/core/???/mqt-catalyst-plugin.dll"))
        if catalyst_plugin_abs_path.exists() and catalyst_plugin_abs_path.is_file():
            return catalyst_plugin_abs_path
        msg = "mqt-catalyst-plugin library not found."
        raise FileNotFoundError(msg)
    except PackageNotFoundError:
        msg = "mqt-core not installed, installation required to access the mqt-catalyst-plugin library."
        raise ImportError(msg) from None


def name2pass(_name: str) -> tuple[Path, str]:
    """Example entry point for MQT plugin."""
    return get_catalyst_plugin_abs_path(), "mqt-core-round-trip"


def MQTCoreRoundTrip(*flags, **valued_options):
    """Applies the "mqt-core-round-trip" pass."""

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
