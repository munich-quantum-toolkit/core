# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from collections.abc import Iterator, Mapping, Sequence
from typing import overload

class Variable:
    """A symbolic variable.

    Note:
        Variables are uniquely identified by their name, so if a variable with the same name already exists, the existing variable will be returned.

    Args:
        name: The name of the variable.
    """

    def __init__(self, name: str = "") -> None: ...
    @property
    def name(self) -> str:
        """The name of the variable."""

    def __eq__(self, arg: object, /) -> bool: ...
    def __ne__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __lt__(self, arg: Variable, /) -> bool: ...
    def __gt__(self, arg: Variable, /) -> bool: ...

class Term:
    """A symbolic term which consists of a variable with a given coefficient.

    Args:
        variable: The variable of the term.
        coefficient: The coefficient of the term.
    """

    def __init__(self, variable: Variable, coefficient: float = 1.0) -> None: ...
    @property
    def variable(self) -> Variable:
        """The variable of the term."""

    @property
    def coefficient(self) -> float:
        """The coefficient of the term."""

    def has_zero_coefficient(self) -> bool:
        """Check if the coefficient of the term is zero."""

    def add_coefficient(self, coeff: float) -> None:
        """Add a coefficient to the coefficient of this term.

        Args:
            coeff: The coefficient to add.
        """

    def evaluate(self, assignment: Mapping[Variable, float]) -> float:
        """Evaluate the term with a given variable assignment.

        Args:
            assignment: The variable assignment.

        Returns:
            The evaluated value of the term.
        """

    def __mul__(self, arg: float, /) -> Term: ...
    def __rmul__(self, arg: float, /) -> Term: ...
    def __truediv__(self, arg: float, /) -> Term: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __ne__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...

class Expression:
    r"""A symbolic expression which consists of a sum of terms and a constant.

    The expression is of the form :math:`constant + term_1 + term_2 + \dots + term_n`.
    Alternatively, an expression can be created with a single term and a constant or just a constant.

    Args:
        terms: The list of terms.
        constant: The constant.
    """

    @overload
    def __init__(self, constant: float = 0.0) -> None: ...
    @overload
    def __init__(self, terms: Sequence[Term], constant: float = 0.0) -> None: ...
    @overload
    def __init__(self, term: Term, constant: float = 0.0) -> None: ...
    @property
    def constant(self) -> float:
        """The constant of the expression."""

    @constant.setter
    def constant(self, arg: float, /) -> None: ...
    def __iter__(self) -> Iterator[Term]: ...
    def __getitem__(self, index: int) -> Term: ...
    def is_zero(self) -> bool:
        """Check if the expression is zero."""

    def is_constant(self) -> bool:
        """Check if the expression is a constant."""

    def num_terms(self) -> int:
        """The number of terms in the expression."""

    def __len__(self) -> int: ...
    @property
    def terms(self) -> list[Term]:
        """The terms of the expression."""

    @property
    def variables(self) -> set[Variable]:
        """The variables in the expression."""

    def evaluate(self, assignment: Mapping[Variable, float]) -> float:
        """Evaluate the expression with a given variable assignment.

        Args:
            assignment: The variable assignment.

        Returns:
            The evaluated value of the expression.
        """

    @overload
    def __add__(self, arg: Expression, /) -> Expression: ...
    @overload
    def __add__(self, arg: float, /) -> Expression: ...
    @overload
    def __add__(self, arg: Term, /) -> Expression: ...
    @overload
    def __radd__(self, arg: Term, /) -> Expression: ...
    @overload
    def __radd__(self, arg: float, /) -> Expression: ...
    @overload
    def __sub__(self, arg: Expression, /) -> Expression: ...
    @overload
    def __sub__(self, arg: float, /) -> Expression: ...
    @overload
    def __sub__(self, arg: Term, /) -> Expression: ...
    @overload
    def __rsub__(self, arg: float, /) -> Expression: ...
    @overload
    def __rsub__(self, arg: Term, /) -> Expression: ...
    def __mul__(self, arg: float, /) -> Expression: ...
    def __rmul__(self, arg: float, /) -> Expression: ...
    def __truediv__(self, arg: float, /) -> Expression: ...
    def __eq__(self, arg: object, /) -> bool: ...
    def __ne__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
