# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Keep operation doubles unconstrained while querying real native metadata."""

from __future__ import annotations

import pytest

from mqt.core.plugins.qiskit import backend
from mqt.core.qdmi import Device


@pytest.fixture
def mock_parameter_constraints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operation doubles model devices without the custom parameter contract."""
    query = backend._operation_parameter_constraints  # ruff: ignore[private-member-access]
    monkeypatch.setattr(
        backend, "_operation_parameter_constraints", lambda op: query(op) if isinstance(op, Device.Operation) else []
    )
