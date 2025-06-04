# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import assert_never
elif TYPE_CHECKING:
    from typing_extensions import assert_never
else:

    def assert_never(_: object) -> None:
        msg = "Expected code to be unreachable"
        raise AssertionError(msg)


__all__ = ["assert_never"]


def __dir__() -> list[str]:
    return __all__
