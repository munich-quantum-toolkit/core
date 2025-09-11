# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MQTOpt xDSL dialects module."""

from __future__ import annotations

from .MQTOpt.MQTOptOps import MQTOpt


def get_all_dialects():
    """Return a dictionary of all available dialects."""

    def get_mqtopt():
        return MQTOpt

    return {
        "mqtopt": get_mqtopt,
    }
