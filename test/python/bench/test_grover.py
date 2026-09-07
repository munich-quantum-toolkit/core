# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Python bindings for the grover benchmark."""

from __future__ import annotations

import pytest

from mqt.core.bench import grover

from .utils import assert_generates


def test_grover_resolves_iterations_and_reports_success() -> None:
    """Expose Grover's resolved default and marked-outcome score."""
    options = grover.Options(marked_bitstring="10")
    benchmark = grover.Grover(options)

    assert options.iterations is None
    assert benchmark.options.iterations == 1
    assert benchmark.qubits == 2
    assert benchmark.probability("10") == pytest.approx(1)
    assert benchmark.evaluate({"10": 20}).success_probability == pytest.approx(1)

    copy = grover.Grover.from_manifest_json(benchmark.manifest_json)
    assert copy.instance_specification_json == benchmark.instance_specification_json
    assert copy.case_id == benchmark.case_id
    assert_generates(benchmark.generate())
