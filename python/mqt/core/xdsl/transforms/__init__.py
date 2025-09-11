# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQT Core xDSL transforms module."""

from __future__ import annotations

from mqt.core.xdsl.transforms.convert_mqtopt_to_qssa import ConvertMQTOptToQssa


def get_all_passes():
    """Return a dictionary of all available passes."""
    return {
        "convert-mqtopt-to-qssa": lambda: ConvertMQTOptToQssa(),
    }
