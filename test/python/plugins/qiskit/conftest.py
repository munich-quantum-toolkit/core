# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Keep Python device doubles at the native snapshot boundary."""

from __future__ import annotations

import pytest

from mqt.core.mlir import CompilerTarget
from mqt.core.qdmi import Device


@pytest.fixture
def mock_device_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Snapshot Python device doubles; genuine devices use the native adapter."""
    native_snapshot = CompilerTarget.from_device

    def snapshot(device: Device) -> CompilerTarget:
        if isinstance(device, Device):
            return native_snapshot(device)
        operations = []
        for operation in device.operations():
            arity = operation.qubits_num()
            placements = operation.site_pairs() if arity == 2 else operation.sites()
            tuples = (
                []
                if placements is None
                else [
                    CompilerTarget.SiteTuple([site.index() for site in pair])
                    for pair in (placements if arity == 2 else [(site,) for site in placements])
                ]
            )
            operations.append(
                CompilerTarget.OperationCapability(
                    operation.name(), arity, operation.parameters_num(), site_tuples=tuples
                )
            )
        coupling = device.coupling_map()
        return CompilerTarget(
            device.name(),
            [CompilerTarget.Site(site.index()) for site in device.sites()],
            connectivity=CompilerTarget.Connectivity.all_to_all()
            if coupling is None
            else CompilerTarget.Connectivity([(a.index(), b.index()) for a, b in coupling]),
            native_operations=CompilerTarget.NativeOperations(operations),
        )

    monkeypatch.setattr(CompilerTarget, "from_device", snapshot)
