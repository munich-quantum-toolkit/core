# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Compatibility imports for typing features."""

import sys

if sys.version_info >= (3, 11):
    from typing import Unpack as _Unpack
else:
    from typing_extensions import Unpack as _Unpack

# Give introspection tools an unconditional module-level export.
Unpack = _Unpack

__all__ = ["Unpack"]


def __dir__() -> list[str]:
    return __all__
