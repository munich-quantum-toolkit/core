# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Advisory warnings for optional integrations."""


class UntestedVersionWarning(UserWarning):
    """An optional integration uses a newer interface than Core has tested."""
