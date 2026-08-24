# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared public typing helpers for MQT Core."""

import os  # ruff: ignore[typing-only-standard-library-import]
from typing import Any, TypedDict

__all__ = ["QDMIJobParameters", "QDMISessionParameters", "QiskitEstimatorOptions", "QiskitSamplerOptions"]


def __dir__() -> list[str]:
    return __all__


class QDMISessionParameters(TypedDict, total=False):
    """Keyword arguments accepted when opening a QDMI device session."""

    driver_path: str | os.PathLike[str] | None
    token: str | None
    auth_file: str | os.PathLike[str] | None
    auth_url: str | None
    username: str | None
    password: str | None
    project_id: str | None
    custom1: str | None
    custom2: str | None
    custom3: str | None
    custom4: str | None
    custom5: str | None


class QDMIJobParameters(TypedDict, total=False):
    """Custom keyword arguments accepted when submitting a QDMI job."""

    custom1: str | bool | float | None
    custom2: str | bool | float | None
    custom3: str | bool | float | None
    custom4: str | bool | float | None
    custom5: str | bool | float | None


class QiskitSamplerOptions(TypedDict, total=False):
    """Native options accepted by ``QDMIBackend.sampler``."""

    default_shots: int
    seed_simulator: int | None
    run_options: dict[str, Any] | None


class QiskitEstimatorOptions(TypedDict, total=False):
    """Native options accepted by ``QDMIBackend.estimator``."""

    default_precision: float
    abelian_grouping: bool
    seed_simulator: int | None
