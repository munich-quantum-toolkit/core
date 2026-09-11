# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared checks for benchmark Python bindings."""

from mqt.core import mlir


def assert_generates(program: mlir.QCProgram) -> None:
    """Exercise the Python-to-MLIR generation boundary."""
    assert isinstance(program.to_qco(), mlir.QCOProgram)
