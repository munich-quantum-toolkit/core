# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Optional dependency management.

Inspired by:
- Qiskit's LazyDependencyManager: https://github.com/Qiskit/qiskit/blob/main/qiskit/utils/lazy_tester.py
"""

from __future__ import annotations

import contextlib
import importlib
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

__all__ = ["OptionalDependencyTester"]


class OptionalDependencyTester:
    """Manager for optional dependencies with lazy testing and caching.

    This class provides a reusable pattern for managing optional dependencies
    across MQT projects. It supports:
    - Lazy evaluation and caching of import checks
    - Boolean context for conditional logic
    - Required imports with custom error messages
    - Test-friendly context manager to temporarily disable availability

    Examples:
        Basic usage with boolean context:

        >>> HAS_QISKIT = OptionalDependencyTester("qiskit")
        >>> if HAS_QISKIT:
        ...     import qiskit
        ...     # Use qiskit

        Requiring with custom message:

        >>> HAS_QISKIT.require_now("use Qiskit backends")  # Raises if not available

        Getting the imported module:

        >>> qiskit = HAS_QISKIT.require_module("use Qiskit functionality")

        Testing with temporary disable:

        >>> with HAS_QISKIT.disable_locally():
        ...     assert not HAS_QISKIT
    """

    def __init__(
        self,
        module: str,
        *,
        install_msg: str | None = None,
        warn_on_fail: bool = False,
    ) -> None:
        """Initialize an optional dependency tester.

        Args:
            module: Name of the module to test (e.g., "qiskit", "matplotlib").
            install_msg: Optional installation instructions for error messages.
                If not provided, defaults to "Install with 'pip install {module}'".
            warn_on_fail: If True, emit a warning when import fails. Default is False
                to avoid noise for truly optional dependencies.
        """
        self._module = module
        self._bool: bool | None = None
        self._install_msg = install_msg or f"Install with 'pip install {module}'"
        self._warn_on_fail = warn_on_fail

    def _is_available(self) -> bool:
        """Test module availability by attempting import.

        Returns:
            True if the module can be imported, False otherwise.
        """
        try:
            importlib.import_module(self._module)
        except ImportError as exc:
            if self._warn_on_fail:  # pragma: no cover
                warnings.warn(
                    f"Optional module '{self._module}' failed to import: {exc!r}",
                    category=ImportWarning,
                    stacklevel=3,
                )
            return False
        else:
            return True

    def __bool__(self) -> bool:
        """Check if the dependency is available (cached).

        Returns:
            True if the dependency is available, False otherwise.
        """
        if self._bool is None:
            self._bool = self._is_available()
        return self._bool

    def require_now(self, feature: str = "use this feature") -> None:
        """Require the dependency is available, raising if not.

        Args:
            feature: Description of what requires this dependency (for error message).

        Raises:
            ImportError: If the dependency is not available.
        """
        if self:
            return
        msg = f"The '{self._module}' library is required to {feature}. {self._install_msg}"
        raise ImportError(msg)

    def require_module(self, feature: str = "use this feature") -> ModuleType:
        """Require the dependency and return the imported module.

        Args:
            feature: Description of what requires this dependency (for error message).

        Returns:
            The imported module.
        """
        self.require_now(feature)
        return importlib.import_module(self._module)

    @contextlib.contextmanager
    def disable_locally(self) -> Generator[None, None, None]:
        """Context manager to temporarily treat dependency as unavailable.

        This is useful for testing code paths that handle missing dependencies.

        Yields:
            None

        Examples:
            >>> HAS_QISKIT = OptionalDependencyTester("qiskit")
            >>> with HAS_QISKIT.disable_locally():
            ...     assert not HAS_QISKIT  # Temporarily False
            >>> assert HAS_QISKIT  # Back to actual state
        """
        previous = self._bool
        self._bool = False
        try:
            yield
        finally:
            self._bool = previous

    @property
    def module_name(self) -> str:
        """Get the module name being tested.

        Returns:
            The module name string.
        """
        return self._module

    def __repr__(self) -> str:
        """Return string representation.

        Returns:
            String representation showing module and availability status.
        """
        available = "available" if self else "not available"
        return f"<OptionalDependencyTester(module={self._module!r}, {available})>"
