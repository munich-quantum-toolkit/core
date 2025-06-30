# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from collections.abc import Iterator, Mapping, Sequence
from typing import overload

__all__ = ["Expression", "Term", "Variable"]

class Variable:
    """A symbolic variable.

    Args:
        name: The name of the variable.

    Note:
        Variables are uniquely identified by their name, so if a variable with the same name already exists,
        the existing variable will be returned.
    """

    def __eq__(self, arg0: object) -> bool: ...
    def __gt__(self, arg0: Variable) -> bool: ...
    def __hash__(self) -> int: ...
    def __init__(self, name: str = "") -> None: ...
    def __lt__(self, arg0: Variable) -> bool: ...
    def __ne__(self, arg0: object) -> bool: ...
    @property
    def name(self) -> str:
        """The name of the variable."""

class Term:
    """A symbolic term which consists of a variable with a given coefficient.

    Args:
        variable: The variable of the term.
        coefficient: The coefficient of the term.
    """

    def __eq__(self, arg0: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __init__(self, variable: Variable, coefficient: float = 1.0) -> None: ...
    def __mul__(self, arg0: float) -> Term: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __rmul__(self, arg0: float) -> Term: ...
    def __rtruediv__(self, arg0: float) -> Term: ...
    def __truediv__(self, arg0: float) -> Term: ...
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

    def has_zero_coefficient(self) -> bool:
        """Check if the coefficient of the term is zero."""

    @property
    def coefficient(self) -> float:
        """The coefficient of the term."""

    @property
    def variable(self) -> Variable:
        """The variable of the term."""

class Expression:
    r"""A symbolic expression which consists of a sum of terms and a constant.

    The expression is of the form :math:`constant + term_1 + term_2 + \dots + term_n`.

    Args:
        terms: The list of terms.
        constant: The constant.

    Alternatively, an expression can be created with a single term and a constant or just a constant.
    """

    constant: float
    """
    The constant of the expression.
    """

    @overload
    def __add__(self, arg0: Expression) -> Expression: ...
    @overload
    def __add__(self, arg0: Term) -> Expression: ...
    @overload
    def __add__(self, arg0: float) -> Expression: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getitem__(self, idx: int) -> Term: ...
    def __hash__(self) -> int: ...
    @overload
    def __init__(self, terms: Sequence[Term], constant: float = 0.0) -> None:
        """Create an expression with a given list of terms and a constant.

        Args:
            terms: The list of terms.
            constant: The constant.
        """

    @overload
    def __init__(self, term: Term, constant: float = 0.0) -> None:
        """Create an expression with a given term and a constant.

        Args:
            term: The term.
            constant: The constant.
        """

    @overload
    def __init__(self, constant: float = 0.0) -> None:
        """Create an expression with a given constant.

        Args:
            constant: The constant.
        """

    def __iter__(self) -> Iterator[Term]: ...
    def __len__(self) -> int: ...
    def __mul__(self, arg0: float) -> Expression: ...
    def __ne__(self, arg0: object) -> bool: ...
    @overload
    def __radd__(self, arg0: Term) -> Expression: ...
    @overload
    def __radd__(self, arg0: float) -> Expression: ...
    def __rmul__(self, arg0: float) -> Expression: ...
    @overload
    def __rsub__(self, arg0: Term) -> Expression: ...
    @overload
    def __rsub__(self, arg0: float) -> Expression: ...
    def __rtruediv__(self, arg0: float) -> Expression: ...
    @overload
    def __sub__(self, arg0: Expression) -> Expression: ...
    @overload
    def __sub__(self, arg0: Term) -> Expression: ...
    @overload
    def __sub__(self, arg0: float) -> Expression: ...
    def __truediv__(self, arg0: float) -> Expression: ...
    def evaluate(self, assignment: Mapping[Variable, float]) -> float:
        """Evaluate the expression with a given variable assignment.

        Args:
            assignment: The variable assignment.

        Returns:
            The evaluated value of the expression.
        """

    def is_constant(self) -> bool:
        """Check if the expression is a constant."""

    def is_zero(self) -> bool:
        """Check if the expression is zero."""

    def num_terms(self) -> int:
        """The number of terms in the expression."""

    @property
    def terms(self) -> list[Term]:
        """The terms of the expression."""

    @property
    def variables(self) -> set[Variable]:
        """The variables in the expression."""
